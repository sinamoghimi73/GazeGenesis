import torch
import torch.nn as nn
from GazeGenesis.Utility.device import get_device_name

class PatchEmbeddings(nn.Module):
    def __init__(self, image_size = 224, patch_size = 16, in_channels = 3, embed_dim = 768):
        super(PatchEmbeddings, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        self.num_patches = int(image_size // patch_size) ** 2
        self.patchifier = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # (batch_size, in_channels, image_size, image_size) -> (batch_size, num_patches, embed_dim)
        x = self.patchifier(x) # (batch_size, embed_dim, num_patches ** 0.5, num_patches ** 0.5)
        x = x.flatten(2) # (batch_size, embed_dim, num_patches)
        x = x.transpose(1, 2) # (batch_size, num_patches, embed_dim) 
        return x
    

class Attention(nn.Module):
    def __init__(self, dim, n_heads, qkv_bias = True, attn_p = 0.1, proj_p = 0.1):
        super(Attention, self).__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)


    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim)  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        attention_scores = torch.matmul(q, k_t) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attention_probs = attention_scores.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attention_probs = self.attn_drop(attention_probs)

        weighted_avg = torch.matmul(attention_probs, v)  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(1, 2)  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p = 0.):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.GELU(),
            nn.Dropout(p),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(p)
        )
    
    def forward(self, x):
        return self.layers(x)
    
class Block(nn.Module):
    def __init__(self, dim, n_heads, mlp_ratio = 4.0, qkv_bias = True, p = 0., attn_p = 0.):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
            dim = dim, n_heads = n_heads, qkv_bias = qkv_bias, attn_p = attn_p, proj_p = p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)

        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features = dim, hidden_features = hidden_features, out_features = dim, p = 0.
        )

    def forward(self, x):
        x += x + self.attn(self.norm1(x))
        x += self.mlp(self.norm2(x))
        return x
    
class VisionTransformer(nn.Module):
    def __init__(self, 
                 img_size = 384, patch_size = 16, in_channels = 3, n_classes = 10, embed_dim = 768, depth = 12, n_heads = 12, mlp_ratio = 4., qkv_bias = True, p = 0., attn_p = 0.
                 ):
        super(VisionTransformer, self).__init__()
        self.patch_embed = PatchEmbeddings(image_size = img_size, patch_size = patch_size, in_channels = in_channels, embed_dim = embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.num_patches, embed_dim))

        self.pos_drop = nn.Dropout(p)
        self.blocks = nn.ModuleList([Block(dim = embed_dim, n_heads = n_heads, mlp_ratio = mlp_ratio, qkv_bias = qkv_bias, p = p, attn_p = attn_p)]*depth)

        self.norm = nn.LayerNorm(embed_dim, eps = 1e-6)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.patch_embed(x)

        cls_token = self.cls_token.expand(batch_size, -1, -1) # (n_samples, 1, embed_dim)
        x = torch.cat((cls_token, x), dim = 1) # (n_samples, 1 + n_patches, embed_dim)

        x += self.pos_embed # (n_samples, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        cls_token_final = x[:, 0] # just the CLS token
        x = self.head(cls_token_final)

        return x



        
if __name__ == "__main__":
    x = torch.randn(1, 3, 28, 28)
    vit = VisionTransformer(
        img_size = 28,
        patch_size= 4,
        in_channels = 3,
        n_classes = 10,
        embed_dim = 8,
        depth = 1,
        n_heads = 4,
        mlp_ratio= 4.,
        qkv_bias= True,
        p = 0.1,
        attn_p= 0.1
    )

    y = vit(x)
    print(y.shape)

    

