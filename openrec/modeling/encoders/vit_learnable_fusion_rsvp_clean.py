"""Clean ViT encoder for two interaction experiments:
1) learnable multi-scale fusion + scaled residual + RSVP
2) learnable multi-scale fusion parallel to RSVP

This file intentionally removes the legacy RDBAFormer branches (RTL/TriZone/Cross/
Patch/Gate), DFE/RCSM/ACF/IRF/ACDI, vertical-pooling variants, and all previous
ablation modes. It keeps only the modules required by the two experiments.
"""

import numpy as np
import torch
from torch import nn
from torch.nn.init import kaiming_normal_, ones_, trunc_normal_, zeros_

from openrec.modeling.common import Block, PatchEmbed


class _DilatedAttention(nn.Module):
    """Spatial mask predictor for one scale."""

    def __init__(self, dim: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")

        padding = dilation * (kernel_size // 2)
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.predictor = nn.Conv2d(dim, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.act(x)
        return torch.sigmoid(self.predictor(x))


class _DilatedEncoder(nn.Module):
    """Depthwise-pointwise feature encoder for one scale."""

    def __init__(self, dim: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError(
                f"kernel_size must be a positive odd integer, got {kernel_size}"
            )
        if dilation <= 0:
            raise ValueError(f"dilation must be positive, got {dilation}")

        padding = dilation * (kernel_size // 2)
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            groups=dim,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()
        self.pointwise = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.norm(x)
        x = self.act(x)
        return self.pointwise(x)


class LearnableMultiScaleEnhancement(nn.Module):
    """Fine/context enhancement with one dataset-level learnable scale pair.

    Input / output correction shape: [B, C, H, W].

    Fine dilation = 1
    Context dilation = 2 by default

    w = softmax([a_f, a_c]), with [a_f, a_c] initialized to [0, 0],
    therefore the initial fusion is exactly 0.5 / 0.5.
    """

    def __init__(
        self,
        dim: int,
        kernel_size: int = 3,
        context_dilation: int = 2,
        drop_rate: float = 0.1,
    ) -> None:
        super().__init__()

        self.fine_attention = _DilatedAttention(
            dim=dim,
            kernel_size=kernel_size,
            dilation=1,
        )
        self.fine_encoder = _DilatedEncoder(
            dim=dim,
            kernel_size=kernel_size,
            dilation=1,
        )

        self.context_attention = _DilatedAttention(
            dim=dim,
            kernel_size=kernel_size,
            dilation=context_dilation,
        )
        self.context_encoder = _DilatedEncoder(
            dim=dim,
            kernel_size=kernel_size,
            dilation=context_dilation,
        )

        # Exactly two global learnable parameters.
        self.scale_logits = nn.Parameter(torch.zeros(2))

        # Same outer projection as the previous clean component-only wrapper:
        # Conv1x1 -> BN -> Dropout, no activation.
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim),
        )
        self.drop = nn.Dropout(drop_rate) if drop_rate > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m_f = self.fine_attention(x)
        f_f = self.fine_encoder(x) * (1.0 + m_f)

        m_c = self.context_attention(x)
        f_c = self.context_encoder(x) * (1.0 + m_c)

        weights = torch.softmax(self.scale_logits, dim=0)
        fused = weights[0] * f_f + weights[1] * f_c

        return self.drop(self.out_proj(fused))


class StaticVerticalPooling(nn.Module):
    """Global learnable height prior shared by sample, width, and channel."""

    def __init__(self, feature_height: int) -> None:
        super().__init__()
        self.height_logits = nn.Parameter(torch.zeros(feature_height))

    def reset_to_uniform(self) -> None:
        with torch.no_grad():
            self.height_logits.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        if x.shape[1] != self.height_logits.numel():
            raise ValueError(
                "StaticVerticalPooling height mismatch: "
                f"{x.shape[1]} != {self.height_logits.numel()}"
            )
        weights = torch.softmax(self.height_logits, dim=0)
        return torch.sum(x * weights.view(1, -1, 1, 1), dim=1)


class ResidualStaticVerticalPooling(nn.Module):
    """RSVP: average pooling plus a learnable static vertical residual."""

    def __init__(self, feature_height: int, init_alpha: float = 0.1) -> None:
        super().__init__()
        if not 0.0 < init_alpha < 1.0:
            raise ValueError(f"init_alpha must be in (0, 1), got {init_alpha}")

        self.static_pool = StaticVerticalPooling(feature_height)
        init_logit = float(np.log(init_alpha / (1.0 - init_alpha)))
        self.alpha_logit = nn.Parameter(
            torch.tensor(init_logit, dtype=torch.float32)
        )

    def reset_to_uniform(self) -> None:
        self.static_pool.reset_to_uniform()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, H, W, C]
        z_avg = x.mean(dim=1)
        z_static = self.static_pool(x)
        alpha = torch.sigmoid(self.alpha_logit)
        return z_avg + alpha * (z_static - z_avg)


class ViTLearnableFusionRSVP(nn.Module):
    """Pure ViT + learnable multi-scale enhancement + RSVP encoder.

    interaction_mode:
      - "scaled_serial":
          F1 = F + gamma * DeltaF, gamma starts at interaction_init.
          RSVP receives F1.

      - "parallel":
          RSVP receives the untouched normalized ViT feature.
          Multi-scale enhancement contributes through an AvgPool residual:
          Z = RSVP(LN(F)) + lambda * AvgPool(LN(F + DeltaF) - LN(F)),
          where lambda starts at interaction_init.

      - "intermediate_rsvp":
          Insert full-strength MSFE after enhancement_after_block Transformer
          blocks, continue the remaining ViT blocks, then apply LN + RSVP.
    """

    def __init__(
        self,
        img_size=(32, 128),
        patch_size=(4, 4),
        in_channels=3,
        out_channels=256,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        last_stage=True,
        feat2d=False,
        component_kernel=3,
        context_dilation=2,
        enhancement_drop_rate=0.1,
        interaction_mode="scaled_serial",
        enhancement_after_block=6,
        interaction_init=0.1,
        rsvp_init_alpha=0.1,
        last_drop=0.1,
        vertical_pooling: str = "rsvp",
        **kwargs,
    ) -> None:
        super().__init__()

        if not last_stage:
            raise ValueError("ViTLearnableFusionRSVP requires last_stage=True")
        if feat2d:
            raise ValueError("ViTLearnableFusionRSVP requires feat2d=False")

        interaction_mode = str(interaction_mode).lower()
        valid_modes = {"baseline", "learnable_only", "rsvp_only", "scaled_serial", "parallel", "intermediate_rsvp"}
        if interaction_mode not in valid_modes:
            raise ValueError(
                f"interaction_mode must be one of {sorted(valid_modes)}, "
                f"got {interaction_mode!r}"
            )
        if not 0.0 < interaction_init < 1.0:
            raise ValueError(
                f"interaction_init must be in (0, 1), got {interaction_init}"
            )

        self.img_size = list(img_size)
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.feat_sz = [
            self.img_size[0] // patch_size[0],
            self.img_size[1] // patch_size[1],
        ]
        self.interaction_mode = interaction_mode
        self.vertical_pooling = str(vertical_pooling).lower()
        valid_pooling = {"uniform", "static", "rsvp"}
        if self.vertical_pooling not in valid_pooling:
            raise ValueError(
                f"vertical_pooling must be one of {sorted(valid_pooling)}, "
                f"got {self.vertical_pooling!r}"
            )

        if (
            self.interaction_mode != "intermediate_rsvp"
            and self.vertical_pooling != "rsvp"
        ):
            raise ValueError(
                "uniform/static pooling is supported only with "
                "interaction_mode='intermediate_rsvp'"
            )
        self.enhancement_after_block = int(enhancement_after_block)
        if not 1 <= self.enhancement_after_block < depth:
            raise ValueError(
                "enhancement_after_block must be in [1, depth-1], "
                f"got {self.enhancement_after_block} for depth={depth}"
            )

        self.patch_embed = PatchEmbed(
            self.img_size,
            patch_size,
            in_channels,
            embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(
            torch.zeros([1, num_patches, embed_dim], dtype=torch.float32)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=act_layer,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Shared final LayerNorm, matching the original ViT pipeline.
        self.norm = norm_layer(embed_dim)

        self.enhancement = LearnableMultiScaleEnhancement(
            dim=embed_dim,
            kernel_size=component_kernel,
            context_dilation=context_dilation,
            drop_rate=enhancement_drop_rate,
        )

        self.rsvp = ResidualStaticVerticalPooling(
            feature_height=self.feat_sz[0],
            init_alpha=rsvp_init_alpha,
        )
        # 三组保留相同模块构造过程，只冻结不参与本组计算的池化参数。
        # 主干、MSFE 和识别头继续正常训练。
        if self.interaction_mode == "intermediate_rsvp":
            if self.vertical_pooling == "uniform":
                self.rsvp.requires_grad_(False)
            elif self.vertical_pooling == "static":
                self.rsvp.alpha_logit.requires_grad_(False)

        init_logit = float(
            np.log(interaction_init / (1.0 - interaction_init))
        )
        self.interaction_logit = nn.Parameter(
            torch.tensor(init_logit, dtype=torch.float32)
        )

        # Same projection as LastStage:
        # Linear(384 -> 256, bias=False) -> Hardswish -> Dropout.
        self.last_conv = nn.Linear(embed_dim, out_channels, bias=False)
        self.hardswish = nn.Hardswish()
        self.last_dropout = nn.Dropout(p=last_drop)

        trunc_normal_(self.pos_embed, mean=0, std=0.02)
        self.apply(self._init_weights)

        # Explicitly restore the intended exact initial states.
        with torch.no_grad():
            self.enhancement.scale_logits.zero_()
        self.rsvp.reset_to_uniform()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            zeros_(module.bias)
            ones_(module.weight)
        elif isinstance(module, nn.Conv2d):
            kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed"}

    def _tokens_to_bchw(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h, w = self.feat_sz
        if n != h * w:
            raise ValueError(
                f"token count mismatch: {n} != {h}*{w}"
            )
        return x.transpose(1, 2).reshape(b, c, h, w)

    def _bchw_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2).contiguous()

    def _tokens_to_bhwc(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        h, w = self.feat_sz
        if n != h * w:
            raise ValueError(
                f"token count mismatch: {n} != {h}*{w}"
            )
        return x.reshape(b, h, w, c)

    def _project_sequence(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, W, C]
        z = self.last_conv(z)
        z = self.hardswish(z)
        return self.last_dropout(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ViT backbone.
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Experiment A/B: insert MSFE inside the ViT stack, then let the
        # remaining Transformer blocks re-integrate the enhanced 2D feature.
        #
        # enhancement_after_block = 6:
        #   Blocks 1-6 -> MSFE -> Blocks 7-12 -> LN -> RSVP
        #
        # enhancement_after_block = 9:
        #   Blocks 1-9 -> MSFE -> Blocks 10-12 -> LN -> RSVP
        if self.interaction_mode == "intermediate_rsvp":
            for block_idx, block in enumerate(self.blocks, start=1):
                x = block(x)

                if block_idx == self.enhancement_after_block:
                    mid_feature = self._tokens_to_bchw(x)
                    delta_feature = self.enhancement(mid_feature)

                    # Keep the original MSFE outer residual at full strength:
                    # F_out = F + DeltaF.
                    mid_feature = mid_feature + delta_feature
                    x = self._bchw_to_tokens(mid_feature)

            x = self.norm(x)
            x_bhwc = self._tokens_to_bhwc(x)

            # 输入：[B, H, W, C]；输出：[B, W, C]。
            # 仅改变最终高度聚合，MSFE 始终位于 Block 6 后。
            if self.vertical_pooling == "uniform":
                z = x_bhwc.mean(dim=1)
            elif self.vertical_pooling == "static":
                z = self.rsvp.static_pool(x_bhwc)
            else:
                z = self.rsvp(x_bhwc)

            return self._project_sequence(z)

        # Existing end-of-backbone modes.
        for block in self.blocks:
            x = block(x)

        base_tokens = x
        base_feature = self._tokens_to_bchw(base_tokens)

        # --------------------------------------------------------------
        # Clean baseline:
        # ViT -> LN -> AvgPool(H) -> Linear -> Hardswish -> Dropout.
        #
        # Important: enhancement and RSVP are still instantiated in __init__
        # exactly as in the other clean modes, but neither is executed here.
        # This preserves the same constructor/RNG consumption and therefore
        # makes the clean baseline a strict same-initialization control.
        # --------------------------------------------------------------
        if self.interaction_mode == "baseline":
            base_norm = self.norm(base_tokens)
            base_bhwc = self._tokens_to_bhwc(base_norm)
            z = base_bhwc.mean(dim=1)
            return self._project_sequence(z)

        # --------------------------------------------------------------
        # Control 1: learnable multi-scale enhancement only.
        # Same module is instantiated as in every other mode, but RSVP is
        # not executed. This reproduces:
        # ViT -> Learnable Fusion -> outer residual -> LN -> AvgPool -> head.
        # --------------------------------------------------------------
        if self.interaction_mode == "learnable_only":
            delta_feature = self.enhancement(base_feature)
            enhanced_feature = base_feature + delta_feature
            enhanced_tokens = self._bchw_to_tokens(enhanced_feature)
            enhanced_tokens = self.norm(enhanced_tokens)
            enhanced_bhwc = self._tokens_to_bhwc(enhanced_tokens)
            z = enhanced_bhwc.mean(dim=1)
            return self._project_sequence(z)

        # --------------------------------------------------------------
        # Control 2: RSVP only.
        # Enhancement is instantiated but deliberately not executed.
        # This reproduces:
        # ViT -> LN -> RSVP -> head.
        # --------------------------------------------------------------
        if self.interaction_mode == "rsvp_only":
            base_norm = self.norm(base_tokens)
            base_bhwc = self._tokens_to_bhwc(base_norm)
            z = self.rsvp(base_bhwc)
            return self._project_sequence(z)

        # The two interaction experiments use the enhancement branch.
        delta_feature = self.enhancement(base_feature)
        interaction = torch.sigmoid(self.interaction_logit)

        if self.interaction_mode == "scaled_serial":
            # F1 = F + gamma * DeltaF
            enhanced_feature = base_feature + interaction * delta_feature
            enhanced_tokens = self._bchw_to_tokens(enhanced_feature)

            # Keep the original ordering: enhancement -> LayerNorm -> RSVP.
            enhanced_tokens = self.norm(enhanced_tokens)
            enhanced_bhwc = self._tokens_to_bhwc(enhanced_tokens)

            z = self.rsvp(enhanced_bhwc)
            return self._project_sequence(z)

        # Parallel mode:
        # RSVP sees the untouched ViT representation.
        base_norm = self.norm(base_tokens)

        # The enhancement branch keeps the original full outer residual
        # F + DeltaF before measuring its normalized correction.
        full_enhanced_feature = base_feature + delta_feature
        full_enhanced_tokens = self._bchw_to_tokens(full_enhanced_feature)
        enhanced_norm = self.norm(full_enhanced_tokens)

        base_bhwc = self._tokens_to_bhwc(base_norm)
        delta_norm_bhwc = self._tokens_to_bhwc(enhanced_norm - base_norm)

        z_rsvp = self.rsvp(base_bhwc)
        z_delta = delta_norm_bhwc.mean(dim=1)

        # Z = RSVP(base) + lambda * AvgPool(normalized enhancement residual)
        z = z_rsvp + interaction * z_delta
        return self._project_sequence(z)
