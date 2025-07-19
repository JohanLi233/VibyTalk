import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm
from einops import rearrange, einsum
from config import get_config, get_inner_size_from_crop_size
from thop import profile


class UIB(nn.Module):
    def __init__(
        self,
        inp,
        oup,
        stride,
        expand_ratio: float,
        kernel_size=5,
        use_extra_dw_conv=False,
    ):
        super().__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.use_extra_dw_conv = use_extra_dw_conv and self.use_res_connect

        hidden_dim = int(round(inp * expand_ratio))

        self.main_path = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm2d(oup),
        )

        if self.use_extra_dw_conv:
            self.extra_dw_conv = nn.Sequential(
                nn.Conv2d(
                    inp,
                    inp,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    groups=inp,
                    bias=False,
                ),
                nn.BatchNorm2d(inp),
                nn.GELU(),
            )

    def forward(self, x):
        main_output = self.main_path(x)
        if self.use_res_connect:
            if self.use_extra_dw_conv:
                residual = self.extra_dw_conv(x)
                return residual + main_output
            else:
                return x + main_output
        else:
            return main_output


class UIB1d(nn.Module):
    def __init__(
        self, inp, oup, stride, expand_ratio, kernel_size=5, use_extra_dw_conv=False
    ):
        super().__init__()
        self.use_res_connect = stride == 1 and inp == oup
        self.use_extra_dw_conv = use_extra_dw_conv and self.use_res_connect

        hidden_dim = int(round(inp * expand_ratio))

        self.main_path = nn.Sequential(
            nn.Conv1d(inp, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, oup, kernel_size=1, bias=False),
            nn.BatchNorm1d(oup),
        )

        if self.use_extra_dw_conv:
            self.extra_dw_conv = nn.Sequential(
                nn.Conv1d(
                    inp,
                    inp,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=kernel_size // 2,
                    groups=inp,
                    bias=False,
                ),
                nn.BatchNorm1d(inp),
                nn.GELU(),
            )

    def forward(self, x):
        main_output = self.main_path(x)
        if self.use_res_connect:
            if self.use_extra_dw_conv:
                residual = self.extra_dw_conv(x)
                return residual + main_output
            else:
                return x + main_output
        else:
            return main_output


class UIBDownsampleBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        kernel_size=5,
        expand_ratio: float = 2,
    ):
        super().__init__()
        self.double_conv = nn.Sequential(
            UIB(
                in_channels,
                out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                use_extra_dw_conv=False,
            ),
            UIB(
                out_channels,
                out_channels,
                stride=1,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                use_extra_dw_conv=True,
            ),
        )

    def forward(self, x):
        return self.double_conv(x)


class CrossAttentionAudioFuser(nn.Module):
    def __init__(self, img_channels, audio_channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = audio_channels // num_heads

        self.q_proj = nn.Linear(img_channels, audio_channels)
        self.k_proj = nn.Linear(audio_channels, audio_channels)
        self.v_proj = nn.Linear(audio_channels, audio_channels)
        self.out_proj = nn.Linear(audio_channels, img_channels)

        self.norm = nn.LayerNorm(img_channels)

    def forward(self, img_feat, audio_feat):
        _, _, H, W = img_feat.shape

        img_seq = rearrange(img_feat, "b c h w -> b (h w) c")

        q = self.q_proj(img_seq)
        k = self.k_proj(audio_feat)
        v = self.v_proj(audio_feat)

        q = rearrange(q, "b q (h d) -> b h q d", h=self.num_heads)
        k = rearrange(k, "b k (h d) -> b h k d", h=self.num_heads)
        v = rearrange(v, "b v (h d) -> b h v d", h=self.num_heads)

        scale = self.head_dim**-0.5
        attn_scores = einsum(q, k, "b h q d, b h k d -> b h q k") * scale
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_output = einsum(attn_weights, v, "b h q k, b h v d -> b h q d")
        attn_output = rearrange(attn_output, "b h q d -> b q (h d)")

        projected_output = self.out_proj(attn_output)
        output_seq = self.norm(img_seq + projected_output)

        output_feat = rearrange(output_seq, "b (h w) c -> b c h w", h=H, w=W)
        return output_feat


class UpBlock(nn.Module):
    """ """

    def __init__(
        self,
        x1_channels,
        x2_channels,
        out_channels,
        kernel_size=5,
        expand_ratio=2.0,
    ):
        super().__init__()
        concatenated_channels = x1_channels + x2_channels
        self.conv = UIBDownsampleBlock(
            concatenated_channels,
            out_channels,
            stride=1,
            kernel_size=kernel_size,
            expand_ratio=expand_ratio,
        )

    def forward(self, x1, x2):
        x1 = F.interpolate(x1, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        normed_x = self.norm1(x)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + attn_output
        x = x + self.ffn(self.norm2(x))
        return x


class AudioEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        audio_feature_dim,
        audio_feature_groups,
        n_layers,
        n_head,
        audio_seq_len,
        uib_layers=3,
        uib_channels=None,
        uib_kernel_size=5,
        uib_expand_ratio=2.0,
        dropout=0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(audio_feature_groups * audio_feature_dim, d_model)
        self.dropout = nn.Dropout(p=dropout)

        if uib_channels is None:
            uib_channels = [d_model] * uib_layers
        assert len(uib_channels) == uib_layers

        self.local_convs = nn.ModuleList()
        in_channels = d_model
        for i, out_channels in enumerate(uib_channels):
            kernel_size = uib_kernel_size if i < 2 else uib_kernel_size + 2
            self.local_convs.append(
                UIB1d(
                    in_channels,
                    out_channels,
                    stride=1,
                    expand_ratio=uib_expand_ratio,
                    kernel_size=kernel_size,
                    use_extra_dw_conv=True,
                )
            )
            in_channels = out_channels

        final_channels = uib_channels[-1]
        self.channel_proj = (
            nn.Linear(final_channels, d_model)
            if final_channels != d_model
            else nn.Identity()
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, audio_seq_len, d_model))

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(dim=d_model, heads=n_head, dropout=dropout)
                for _ in range(n_layers)
            ]
        )
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        B, T, G, D_feat = x.shape
        x = rearrange(x, "b t g d -> (b t) (g d)")
        x = self.input_proj(x)
        x = rearrange(x, "(b t) c -> b t c", b=B)

        x_conv = rearrange(x, "b t c -> b c t")
        for uib_layer in self.local_convs:
            x_conv = uib_layer(x_conv)
        x = rearrange(x_conv, "b c t -> b t c")
        x = self.channel_proj(x)

        T_seq = x.size(1)
        x = x + self.pos_embedding[:, :T_seq, :]
        x = self.dropout(x)

        outputs = []
        for block in self.transformer_blocks:
            x = block(x)
            outputs.append(x)

        outputs[-1] = self.norm(outputs[-1])

        return tuple(outputs)


class Encoder(nn.Module):
    def __init__(self, n_channels, config):
        super().__init__()
        ch = config.layer_channels
        uib_expand_ratio = config.expansion_factor

        self.inc = UIB(
            n_channels, ch[0], stride=1, expand_ratio=uib_expand_ratio, kernel_size=3
        )
        self.down1 = UIBDownsampleBlock(
            ch[0], ch[1], stride=2, kernel_size=5, expand_ratio=uib_expand_ratio
        )
        self.down2 = UIBDownsampleBlock(
            ch[1], ch[2], stride=2, kernel_size=7, expand_ratio=uib_expand_ratio
        )
        self.down3 = UIBDownsampleBlock(
            ch[2], ch[3], stride=2, kernel_size=7, expand_ratio=uib_expand_ratio
        )
        self.down4 = UIBDownsampleBlock(
            ch[3], ch[4], stride=2, kernel_size=7, expand_ratio=uib_expand_ratio
        )

    def forward(self, img_in):
        x1 = self.inc(img_in)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5


class UNet(nn.Module):
    def __init__(
        self,
        n_channels: int = 6,
        model_size: str = "medium",
    ):
        super().__init__()
        config = get_config(model_size)
        ch = config.layer_channels
        self.model_size = model_size
        self.config = config

        self.audio_encoder_layers = config.transformer_depth
        audio_d_model = ch[4]
        bottleneck_channels = ch[4]

        self.audio_model = AudioEncoder(
            d_model=audio_d_model,
            audio_feature_dim=config.acoustic_vector_size,
            audio_feature_groups=config.acoustic_group_count,
            n_layers=config.transformer_depth,
            n_head=config.attention_head_count,
            audio_seq_len=config.temporal_window_length,
            uib_layers=config.convolution_stack_depth,
            uib_channels=config.conv_channel_progression,
            uib_kernel_size=config.audio_conv_kernel_size,
            uib_expand_ratio=config.audio_expansion_coefficient,
        )
        self.main_encoder = Encoder(n_channels=n_channels, config=config)
        self.audio_fuser = CrossAttentionAudioFuser(
            img_channels=bottleneck_channels,
            audio_channels=audio_d_model,
            num_heads=config.attention_head_count,
        )

        uib_expand_ratio = config.expansion_factor
        self.up1 = UpBlock(
            ch[4], ch[3], ch[3], kernel_size=7, expand_ratio=uib_expand_ratio
        )
        self.up2 = UpBlock(
            ch[3], ch[2], ch[2], kernel_size=7, expand_ratio=uib_expand_ratio
        )
        self.up3 = UpBlock(
            ch[2], ch[1], ch[1], kernel_size=5, expand_ratio=uib_expand_ratio
        )
        self.up4 = UpBlock(
            ch[1], ch[0], ch[0], kernel_size=5, expand_ratio=uib_expand_ratio
        )

        self.outc = OutConv(ch[0], 3)

    def forward(self, img_in, audio_feat):
        x1, x2, x3, x4, x5 = self.main_encoder(img_in)
        a_seq_list = self.audio_model(audio_feat)
        audio_context_seq = a_seq_list[-1]
        fused_bottleneck = self.audio_fuser(x5, audio_context_seq)
        x = self.up1(fused_bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = F.sigmoid(self.outc(x))

        return out


if __name__ == "__main__":
    for model_size in ["nano", "tiny", "base"]:
        print(f"\n{'=' * 20}")
        print(f"Calculating model size: {model_size.upper()}")
        print(f"{'=' * 20}")
        config = get_config(model_size)
        crop_size = config.input_resolution
        inner_size = get_inner_size_from_crop_size(crop_size)
        B = 1
        D_feat = config.acoustic_vector_size
        T_len = config.temporal_window_length
        img_input = torch.randn(B, 6, inner_size, inner_size)
        audio_input = torch.randn(
            B, T_len, config.acoustic_group_count, config.acoustic_vector_size
        )
        net = UNet(n_channels=6, model_size=model_size)
        net.eval()
        inputs = (img_input, audio_input)
        result = profile(net, inputs=inputs, verbose=False)
        macs, params = result[0], result[1]
        gflops = (macs * 2) / 1e9
        mflops = gflops * 1000
        audio_encoder_params = sum(
            p.numel() for p in net.audio_model.parameters() if p.requires_grad
        )
        print(f"Input img shape: {img_input.shape}")
        print(f"Input audio shape: {audio_input.shape}")
        print(f"Total parameters: {params / 1e6:.2f} M")
        print(f"AudioEncoder parameters: {audio_encoder_params / 1e6:.2f} M")
        print(f"MACs: {macs / 1e9:.2f} G")
        print(f"FLOPs: {gflops:.2f} G")
        print(f"FLOPs: {mflops:.2f} M")
