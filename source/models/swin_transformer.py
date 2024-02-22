from torchvision.models import SwinTransformer
from einops import rearrange
import torch


class MySwinTransformer(SwinTransformer):
    def __init__(self, patch_size, embed_dim, depths, num_heads, window_size, **kwargs):
        super(MySwinTransformer, self).__init__(patch_size, embed_dim, depths, num_heads, window_size, **kwargs)
        # Cancel the prediction head
        self.head = torch.nn.Identity()
        self.num_features = embed_dim * 2 ** (len(depths) - 1)

    def forward(self, x):
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x_avg = self.avgpool(x).squeeze()
        x = rearrange(x, 'b d h w -> b (h w) d')

        # Concatenate the pseudo [CLS]
        x = torch.cat([x_avg[:, None, :], x], dim=1)
        return x_avg, x, x


def swin_tiny_window7(**kwargs):
    model = MySwinTransformer(
        patch_size=[4, 4],
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=[7, 7],
        stochastic_depth_prob=0.2,
        **kwargs
    )
    return model


def swin_base_window14(**kwargs):
    model = MySwinTransformer(
        patch_size=[4, 4],
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        window_size=[7, 7],
        stochastic_depth_prob=0.5,
        **kwargs
    )
    return model


if __name__ == '__main__':
    model = swin_tiny_window7().cuda()
    x = torch.randn([10, 3, 224, 224]).cuda()
    y = model(x)
    print(y.shape)
