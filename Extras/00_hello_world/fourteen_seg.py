class Segment:
    def __init__(self, offset=10, spacing=50):
        self.offset = offset
        self.inter_seg_offset = 10
        self.space = spacing

        self.activation_map = {
            "seg1": self.seg1,
            "seg2": self.seg2,
            "seg3": self.seg3,
            "seg4": self.seg4,
            "seg5": self.seg5,
            "seg6": self.seg6,
            "seg7": self.seg7,
            "seg8": self.seg8,
            "seg9": self.seg9,
            "seg10": self.seg10,
            "seg11": self.seg11,
            "seg12": self.seg12,
            "seg13": self.seg13,
            "seg14": self.seg14,
        }

        self.letter_segment_map = {
            "A": [1, 2, 3, 5, 6, 13, 14],
            "B": [1, 2, 3, 4, 8, 11, 14],
            "C": [1, 4, 5, 6],
            "D": [1, 2, 3, 4, 8, 11],
            "E": [1, 4, 5, 6, 13, 14],
            "F": [1, 5, 6, 13, 14],
            "G": [1, 3, 4, 5, 6, 14],
            "H": [2, 3, 5, 6, 13, 14],
            "I": [1, 4, 8, 11],
            "J": [2, 3, 4, 5],
            "K": [5, 6, 9, 12, 13],
            "L": [4, 5, 6],
            "M": [2, 3, 5, 6, 7, 9],
            "N": [2, 3, 5, 6, 7, 12],
            "O": [1, 2, 3, 4, 5, 6],
            "P": [1, 2, 5, 6, 13, 14],
            "Q": [1, 2, 3, 4, 5, 6, 12],
            "R": [1, 2, 5, 6, 12, 13, 14],
            "S": [1, 3, 4, 6, 13, 14],
            "T": [1, 8, 11],
            "U": [2, 3, 4, 5, 6],
            "V": [5, 6, 9, 10],
            "W": [2, 3, 5, 6, 10, 12],
            "X": [7, 9, 10, 12],
            "Y": [7, 9, 11],
            "Z": [1, 4, 9, 10],
        }

    def seg1(self, x1, y1, w, h):
        return [
            x1 + self.offset + self.inter_seg_offset,
            y1 + self.offset - self.inter_seg_offset,
            x1 + w - self.offset - self.inter_seg_offset,
            y1 + self.offset - self.inter_seg_offset,
        ]

    def seg2(self, x1, y1, w, h):
        return [
            x1 + w - self.offset + self.inter_seg_offset,
            y1 + self.offset + self.inter_seg_offset,
            x1 + w - self.offset + self.inter_seg_offset,
            y1 + h // 2 - self.offset // 2 - self.inter_seg_offset,
        ]

    def seg3(self, x1, y1, w, h):
        return [
            x1 + w - self.offset + self.inter_seg_offset,
            y1 + h // 2 + self.offset // 2 + self.inter_seg_offset,
            x1 + w - self.offset + self.inter_seg_offset,
            y1 + h - self.offset - self.inter_seg_offset,
        ]

    def seg4(self, x1, y1, w, h):
        return [
            x1 + self.offset + self.inter_seg_offset,
            y1 + h - self.offset + self.inter_seg_offset,
            x1 + w - self.offset - self.inter_seg_offset,
            y1 + h - self.offset + self.inter_seg_offset,
        ]

    def seg5(self, x1, y1, w, h):
        return [
            x1 + self.offset - self.inter_seg_offset,
            y1 + h // 2 + self.offset // 2 + self.inter_seg_offset,
            x1 + self.offset - self.inter_seg_offset,
            y1 + h - self.offset - self.inter_seg_offset,
        ]

    def seg6(self, x1, y1, w, h):
        return [
            x1 + self.offset - self.inter_seg_offset,
            y1 + self.offset + self.inter_seg_offset,
            x1 + self.offset - self.inter_seg_offset,
            y1 + h // 2 - self.offset // 2 - self.inter_seg_offset,
        ]

    def seg7(self, x1, y1, w, h):
        return [
            x1 + self.offset + self.inter_seg_offset,
            y1 + self.offset + self.inter_seg_offset,
            x1 + w // 2 - self.offset // 2 - self.inter_seg_offset,
            y1 + h // 2 - self.offset // 2 - self.inter_seg_offset,
        ]

    def seg8(self, x1, y1, w, h):
        return [
            x1 + w // 2,
            y1 + self.offset + self.inter_seg_offset,
            x1 + w // 2,
            y1 + h // 2 - self.offset // 2 - self.inter_seg_offset,
        ]

    def seg9(self, x1, y1, w, h):
        return [
            x1 + w - self.offset - self.inter_seg_offset,
            y1 + self.offset + self.inter_seg_offset,
            x1 + w // 2 + self.offset // 2 + self.inter_seg_offset,
            y1 + h // 2 - self.offset // 2 - self.inter_seg_offset,
        ]

    def seg10(self, x1, y1, w, h):
        return [
            x1 + self.offset + self.inter_seg_offset,
            y1 + h - self.offset - self.inter_seg_offset,
            x1 + w // 2 - self.offset // 2 - self.inter_seg_offset,
            y1 + h // 2 + self.offset // 2 + self.inter_seg_offset,
        ]

    def seg11(self, x1, y1, w, h):
        return [
            x1 + w // 2,
            y1 + h // 2 + self.offset // 2 + self.inter_seg_offset,
            x1 + w // 2,
            y1 + h - self.offset - self.inter_seg_offset,
        ]

    def seg12(self, x1, y1, w, h):
        return [
            x1 + w // 2 + self.offset // 2 + self.inter_seg_offset,
            y1 + h // 2 + self.offset // 2 + self.inter_seg_offset,
            x1 + w - self.offset - self.inter_seg_offset,
            y1 + h - self.offset - self.inter_seg_offset,
        ]

    def seg13(self, x1, y1, w, h):
        return [
            x1 + self.offset + self.inter_seg_offset,
            y1 + h // 2,
            x1 + w // 2 - self.offset // 2 + self.inter_seg_offset // 2,
            y1 + h // 2,
        ]

    def seg14(self, x1, y1, w, h):
        return [
            x1 + w // 2 + self.offset // 2 - self.inter_seg_offset // 2,
            y1 + h // 2,
            x1 + w - self.offset - self.inter_seg_offset,
            y1 + h // 2,
        ]

    def apply(self, start_x, start_y, seg_width, seg_height, text):
        arr = []
        for t in text:
            if t == " ":
                start_x += seg_width + self.space
            else:
                segs = [
                    self.activation_map[f"seg{i}"](
                        start_x, start_y, seg_width, seg_height
                    )
                    for i in self.letter_segment_map[t.upper()]
                ]
                arr.extend(segs)
                start_x += seg_width + self.space
        return arr
