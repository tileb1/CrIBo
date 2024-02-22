from source.models.perturbed_topk import PerturbedTopK
from source.models.vision_transformer import Block
from source.utils.utils import trunc_normal_
from einops import rearrange
from torch import nn
import torch


class TokenSampler(nn.Module):
    """ Vision Transformer """

    def __init__(self, embed_dim=768, depth=2, num_heads=1, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, n_tokens=50, **kwargs):
        super().__init__()
        self.n_tokens = n_tokens
        self.topk = PerturbedTopK(k=n_tokens)
        self.uses_hard_topk = True
        self.num_features = self.embed_dim = embed_dim

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                no_mlp=(i == depth - 1))
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Initialize the weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                attention = blk(x, return_attention=True)
        # Extract the distribution from the self-attention
        distribution = attention[:, :, 0, 1:].squeeze()

        # Get the soft indicators
        soft_indicators = self.topk(distribution)

        # Convert from soft to hard
        if self.uses_hard_topk:
            _, k, _ = soft_indicators.shape
            hard_indicators = rearrange(soft_indicators.detach().clone(), 'b k n -> (b k) n')
            max_indices = torch.argmax(hard_indicators, dim=-1)
            hard_indicators = torch.eye(hard_indicators.shape[-1])[:, max_indices]
            hard_indicators = rearrange(hard_indicators, 'n (b k) -> b k n', k=k)
            hard_indicators = hard_indicators.to(soft_indicators.device)
            hard_indicators = hard_indicators + soft_indicators - soft_indicators.detach()
            indicators = hard_indicators
        else:
            indicators = soft_indicators
        return indicators, distribution
