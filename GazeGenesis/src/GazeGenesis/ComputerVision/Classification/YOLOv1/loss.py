#!/usr/bin/python3

import torch
import torch.nn as nn
from GazeGenesis.ComputerVision.Classification.YOLOv1.utils import intersection_over_union


class Loss(nn.Module):
    def __init__(self, num_cells: int = 7, num_boxes: int = 2, num_classes: int = 10):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduce="sum")

        self.num_cells = num_cells
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, prediction, target):
        prediction = prediction.reshape(-1, self.num_cells, self.num_cells, self.num_classes + self.num_boxes * 5)

        iou_b1 = intersection_over_union(prediction[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(prediction[..., 26:30], target[..., 21:25])

        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim = 0)

        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[...,20].unsqueeze(3) # identity of object i -> Iobj_i




        ######## For Box coordinates ########
        box_predictions = exists_box * (
            best_box * prediction[..., 26:30] + (1 - best_box) * prediction[..., 21:25]
        )

        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))


        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        # (N, num_cells, num_cells, 4) -> (N * num_cells * num_cells, 4)
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim = -2),
            torch.flatten(box_targets, end_dim = -2),
        )


        ######## For Object loss ########
        pred_box = (best_box * prediction[..., 25:26]) + ((1 - best_box) * prediction[..., 20:21])

        # (N * num_cells * num_cells)
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box), 
            torch.flatten(exists_box * target[..., 20:21]),
        )

        # (N, num_cells, num_cells, 1) -> # (N, num_cells * num_cells)
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * prediction[..., 20:21], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim = 1),
        )

        no_obj_loss += self.mse(
            torch.flatten((1 - exists_box) * prediction[..., 25:26], start_dim = 1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim = 1),
        )


        ######## For Class loss ########
        # (N, num_cells, num_cells, 20) -> (N * num_cells * num_cells, 20)
        class_loss = self.mse(
            torch.flatten(exists_box * prediction[..., :20], end_dim = -2),
            torch.flatten(exists_box * target[..., :20], end_dim = -2)
        )


        ######## ACTUAL LOSS ########
        loss = (
            self.lambda_coord * box_loss
            + object_loss
            + self.lambda_noobj * no_object_loss
            + class_loss
        )

        return loss