from typing import Any, List, Tuple, Optional, Union, Dict
from einops import rearrange

import json  # [debug-only] text-token norm logging
import os  # [debug-only] text-token norm logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

from .activation_layers import get_activation_layer
from .norm_layers import get_norm_layer
from .embed_layers import TimestepEmbedder, PatchEmbed, TextProjection
from .attenion import attention, parallel_attention, get_cu_seqlens
from .posemb_layers import apply_rotary_emb
from .mlp_layers import MLP, MLPEmbedder, FinalLayer
from .modulate_layers import ModulateDiT, modulate, apply_gate
from .token_refiner import SingleTokenRefiner


class MMDoubleStreamBlock(nn.Module):
    """
    A multimodal dit block with seperate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.img_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.img_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.img_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.txt_attn_qkv = nn.Linear(
            hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs
        )
        self.txt_attn_q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.txt_attn_proj = nn.Linear(
            hidden_size, hidden_size, bias=qkv_bias, **factory_kwargs
        )

        self.txt_norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: tuple = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(
            img_modulated, shift=img_mod1_shift, scale=img_mod1_scale
        )
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(
            img_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(
            txt_modulated, shift=txt_mod1_shift, scale=txt_mod1_scale
        )
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(
            txt_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num
        )
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # Run actual attention.
        q = torch.cat((img_q, txt_q), dim=1)
        k = torch.cat((img_k, txt_k), dim=1)
        v = torch.cat((img_v, txt_v), dim=1)
        assert (
            cu_seqlens_q.shape[0] == 2 * img.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, img.shape[0]:{img.shape[0]}"
        
        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=img_k.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv
            )
            
        # attention computation end

        img_attn, txt_attn = attn[:, : img.shape[1]], attn[:, img.shape[1] :]

        # Calculate the img bloks.
        img = img + apply_gate(self.img_attn_proj(img_attn), gate=img_mod1_gate)
        img = img + apply_gate(
            self.img_mlp(
                modulate(
                    self.img_norm2(img), shift=img_mod2_shift, scale=img_mod2_scale
                )
            ),
            gate=img_mod2_gate,
        )

        # Calculate the txt bloks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn), gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(
                    self.txt_norm2(txt), shift=txt_mod2_shift, scale=txt_mod2_scale
                )
            ),
            gate=txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlock(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim ** -0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(
            hidden_size, hidden_size * 3 + mlp_hidden_dim, **factory_kwargs
        )
        # proj and mlp_out
        self.linear2 = nn.Linear(
            hidden_size + mlp_hidden_dim, hidden_size, **factory_kwargs
        )

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            qk_norm_layer(head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
            if qk_norm
            else nn.Identity()
        )

        self.pre_norm = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6, **factory_kwargs
        )

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_kv: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_kv: Optional[int] = None,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(
            self.linear1(x_mod), [3 * self.hidden_size, self.mlp_hidden_dim], dim=-1
        )

        q, k, v = rearrange(qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        # Apply RoPE if needed.
        if freqs_cis is not None:
            img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
            img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
            img_qq, img_kk = apply_rotary_emb(img_q, img_k, freqs_cis, head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk
            q = torch.cat((img_q, txt_q), dim=1)
            k = torch.cat((img_k, txt_k), dim=1)

        # Compute attention.
        assert (
            cu_seqlens_q.shape[0] == 2 * x.shape[0] + 1
        ), f"cu_seqlens_q.shape:{cu_seqlens_q.shape}, x.shape[0]:{x.shape[0]}"
        
        # attention computation start
        if not self.hybrid_seq_parallel_attn:
            attn = attention(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_kv=max_seqlen_kv,
                batch_size=x.shape[0],
            )
        else:
            attn = parallel_attention(
                self.hybrid_seq_parallel_attn,
                q,
                k,
                v,
                img_q_len=img_q.shape[1],
                img_kv_len=img_k.shape[1],
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_kv=cu_seqlens_kv
            )
        # attention computation end

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        return x + apply_gate(output, gate=mod_gate)


class HYVideoDiffusionTransformer(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    @register_to_config
    def __init__(
        self,
        args: Any,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list

        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        self.text_states_dim = args.text_states_dim
        self.text_states_dim_2 = args.text_states_dim_2

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}"
            )
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(
            self.patch_size, self.in_channels, self.hidden_size, **factory_kwargs
        )

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.text_states_dim, hidden_size, heads_num, depth=2, **factory_kwargs
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # time modulation
        self.time_in = TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs
        )

        # text modulation
        self.vector_in = MLPEmbedder(
            self.text_states_dim_2, self.hidden_size, **factory_kwargs
        )

        # guidance modulation
        self.guidance_in = (
            TimestepEmbedder(
                self.hidden_size, get_activation_layer("silu"), **factory_kwargs
            )
            if guidance_embed
            else None
        )

        # double blocks
        self.double_blocks = nn.ModuleList(
            [
                MMDoubleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    qkv_bias=qkv_bias,
                    **factory_kwargs,
                )
                for _ in range(mm_double_blocks_depth)
            ]
        )

        # single blocks
        self.single_blocks = nn.ModuleList(
            [
                MMSingleStreamBlock(
                    self.hidden_size,
                    self.heads_num,
                    mlp_width_ratio=mlp_width_ratio,
                    mlp_act_type=mlp_act_type,
                    qk_norm=qk_norm,
                    qk_norm_type=qk_norm_type,
                    **factory_kwargs,
                )
                for _ in range(mm_single_blocks_depth)
            ]
        )

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    # ------------------------------------------------------------------ #
    # [debug-only] Text-token norm logging.
    # Append one JSON line per denoising step with the per-token L2 norm of the
    # text tokens before vs after the dual-stream blocks (right before the
    # single-stream merge). Enabled with HUNYUAN_DEBUG_TEXT_NORM=1; output path
    # via HUNYUAN_DEBUG_TEXT_NORM_PATH (default text_norm_debug.jsonl). Saves only
    # per-token scalar norms (256 floats), never full embeddings, and never alters
    # generation. To remove this feature entirely, delete this method, its call
    # site in forward(), the `txt_before_dual` snapshot, and the json/os imports.
    # ------------------------------------------------------------------ #
    def _log_text_token_norm(self, txt_before_dual, txt_after_dual, t):
        # Only rank 0 writes when running distributed (multi-GPU).
        import torch.distributed as dist

        is_rank0 = (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        )
        if not is_rank0:
            return

        path = os.environ.get(
            "HUNYUAN_DEBUG_TEXT_NORM_PATH", "text_norm_debug.jsonl"
        )

        with torch.no_grad():
            # Batch size 1 is the primary use case: log the first batch element.
            before = txt_before_dual[0].float().norm(dim=-1)  # [txt_seq_len]
            after = txt_after_dual[0].float().norm(dim=-1)  # [txt_seq_len]
            ratio = after / before.clamp_min(1e-8)

            # Per-module step counter (lazily initialized so __init__ stays clean).
            step_idx = getattr(self, "_dbg_text_norm_step", 0)
            self._dbg_text_norm_step = step_idx + 1

            try:
                timestep = float(t.flatten()[0].item())
            except Exception:
                timestep = None

            record = {
                "debug_stage": "text_token_norm",
                "step_idx": int(step_idx),
                "timestep": timestep,
                "txt_seq_len": int(after.shape[0]),
                "hidden_dim": int(txt_after_dual.shape[-1]),
                "txt_before_dual_norm": before.tolist(),
                "txt_after_dual_norm": after.tolist(),
                "txt_after_before_ratio": ratio.tolist(),
            }

        debug_dir = os.path.dirname(path)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ #
    # [debug-only] Text-token boundary ablation.
    # Zero real / padding / all text tokens right after the dual-stream blocks
    # (before single-stream), to test whether padding positions act as memory
    # slots. Real tokens are [0, real_end); padding is [real_end, txt_seq_len).
    # Controlled by env vars:
    #   HUNYUAN_TEXT_TOKEN_ABLATION_MODE = none | zero_real_tokens |
    #                                      zero_pad_tokens | zero_all_text
    #   HUNYUAN_TEXT_REAL_TOKEN_END      = real-token count (default 9)
    #   HUNYUAN_TEXT_TOKEN_ABLATION_PATH = JSONL log path (optional)
    #   HUNYUAN_TEXT_TOKEN_ABLATION_PROMPT = prompt string for the log (optional)
    # Deterministic and identical across ranks (text is replicated under
    # sequence parallelism), and a no-op when the mode env var is unset.
    # Easy to remove: delete this method and its call site in forward().
    # ------------------------------------------------------------------ #
    def _apply_text_token_ablation(self, txt, mode, t, img_seq_len):
        seq_len = txt.shape[1]
        real_end = int(os.environ.get("HUNYUAN_TEXT_REAL_TOKEN_END", "9"))
        real_end = max(0, min(real_end, seq_len))

        def _mean_norm(x):
            if x.shape[1] == 0:
                return 0.0
            return x.float().norm(dim=-1).mean().item()

        with torch.no_grad():
            before_txt = _mean_norm(txt)
            before_real = _mean_norm(txt[:, :real_end])
            before_pad = _mean_norm(txt[:, real_end:])

        # Clone so we never mutate a tensor shared elsewhere; apply the ablation.
        txt = txt.clone()
        if mode == "zero_real_tokens":
            txt[:, :real_end] = 0
        elif mode == "zero_pad_tokens":
            txt[:, real_end:] = 0
        elif mode == "zero_all_text":
            txt[:] = 0
        # Unknown modes fall through unchanged (defensive).

        with torch.no_grad():
            after_txt = _mean_norm(txt)
            after_real = _mean_norm(txt[:, :real_end])
            after_pad = _mean_norm(txt[:, real_end:])

        self._log_text_token_ablation(
            mode, real_end, seq_len, img_seq_len, t,
            before_txt, after_txt, before_real, after_real, before_pad, after_pad,
        )
        return txt

    def _log_text_token_ablation(
        self, mode, real_end, txt_seq_len, img_seq_len, t,
        before_txt, after_txt, before_real, after_real, before_pad, after_pad,
    ):
        # rank 0 only when distributed.
        import torch.distributed as dist

        is_rank0 = (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        )
        if not is_rank0:
            return

        try:
            timestep = float(t.flatten()[0].item())
        except Exception:
            timestep = None

        record = {
            "debug_stage": "text_token_ablation",
            "prompt": os.environ.get("HUNYUAN_TEXT_TOKEN_ABLATION_PROMPT"),
            "ablation_mode": mode,
            "timestep": timestep,
            "real_token_range": [0, int(real_end)],
            "pad_token_range": [int(real_end), int(txt_seq_len)],
            "txt_seq_len": int(txt_seq_len),
            "img_seq_len": int(img_seq_len),
            "before_ablation_txt_norm_mean": before_txt,
            "after_ablation_txt_norm_mean": after_txt,
            "real_token_norm_mean_before_ablation": before_real,
            "real_token_norm_mean_after_ablation": after_real,
            "pad_token_norm_mean_before_ablation": before_pad,
            "pad_token_norm_mean_after_ablation": after_pad,
        }

        path = os.environ.get(
            "HUNYUAN_TEXT_TOKEN_ABLATION_PATH", "text_token_ablation_debug.jsonl"
        )
        debug_dir = os.path.dirname(path)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    @staticmethod
    def _dbg_is_rank0():
        # [debug-only] True unless we are a non-zero rank in a distributed run.
        import torch.distributed as dist

        return (
            not dist.is_available()
            or not dist.is_initialized()
            or dist.get_rank() == 0
        )

    # ------------------------------------------------------------------ #
    # [debug-only] Semantic anchor token expansion.
    # After the final dual-stream block, copy clean pre-dual real text tokens
    # (or post-dual real tokens) into the first unused padding slots, then make
    # them visible to the single-stream attention by recomputing cu_seqlens from
    # an updated text mask. Real tokens [0, real_end); anchors
    # [real_end, real_end + repeat*real_end); padding after. Controlled by:
    #   HUNYUAN_TEXT_ANCHOR_EXPANSION_MODE = none | duplicate_before_dual |
    #       duplicate_before_dual_normmatch | duplicate_after_dual
    #   HUNYUAN_TEXT_REAL_TOKEN_END (default 9), HUNYUAN_TEXT_ANCHOR_REPEAT (1),
    #   HUNYUAN_TEXT_ANCHOR_ALPHA (1.0), HUNYUAN_TEXT_ANCHOR_DEBUG_PATH,
    #   HUNYUAN_TEXT_ANCHOR_PROMPT, HUNYUAN_TEXT_ANCHOR_TOKENS_JSON (optional
    #   token_texts for the first-step inspection). Deterministic across ranks.
    # Easy to remove: delete these methods and the call site in forward().
    # ------------------------------------------------------------------ #
    def _apply_text_anchor_expansion(
        self, txt_after_dual, txt_before_dual, text_mask, mode, t,
        img_seq_len, txt_seq_len, cu_seqlens_q_orig,
    ):
        seq_len = txt_after_dual.shape[1]
        real_end = int(os.environ.get("HUNYUAN_TEXT_REAL_TOKEN_END", "9"))
        repeat = int(os.environ.get("HUNYUAN_TEXT_ANCHOR_REPEAT", "1"))
        alpha = float(os.environ.get("HUNYUAN_TEXT_ANCHOR_ALPHA", "1.0"))
        real_end = max(0, min(real_end, seq_len))
        real_len = real_end  # anchor block = copy of the real tokens
        anchor_start = real_end
        anchor_end = real_end + repeat * real_len

        # Safety: skip if there is no room (or nothing to copy).
        if real_len == 0 or repeat < 1 or anchor_end > seq_len:
            if not getattr(self, "_dbg_anchor_warned", False):
                print(
                    f"[anchor_expansion] skip: real_end={real_end} repeat={repeat} "
                    f"-> anchor_end={anchor_end} > txt_seq_len={seq_len}; "
                    f"prompt too long for expansion."
                )
                self._dbg_anchor_warned = True
            return txt_after_dual, cu_seqlens_q_orig

        def _rms(x):
            return x.float().pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()

        def _mean_norm(x):
            if x.shape[1] == 0:
                return 0.0
            return x.float().norm(dim=-1).mean().item()

        after_real = txt_after_dual[:, :real_end]
        real_norm_after_dual = _mean_norm(after_real)

        # Build one anchor block of length real_len.
        if mode == "duplicate_after_dual":
            anchor = alpha * after_real
        elif mode == "duplicate_before_dual":
            anchor = txt_before_dual[:, :real_end]
        elif mode == "duplicate_before_dual_normmatch":
            before_real = txt_before_dual[:, :real_end]
            scale = _rms(after_real) / _rms(before_real)
            anchor = alpha * (before_real.float() * scale)
        else:
            return txt_after_dual, cu_seqlens_q_orig
        anchor = anchor.to(txt_after_dual.dtype)

        # Write anchors into the padding region (clone to avoid shared mutation).
        txt = txt_after_dual.clone()
        tiled = anchor.repeat(1, repeat, 1) if repeat > 1 else anchor
        txt[:, anchor_start:anchor_end] = tiled

        # Make the anchor tokens visible to single-stream by recomputing cu_seqlens
        # from an updated mask: [0, anchor_end) visible, the rest padding.
        new_mask = text_mask.clone()
        new_mask[:, :anchor_end] = 1
        new_mask[:, anchor_end:] = 0
        cu_seqlens_q_single = get_cu_seqlens(new_mask, img_seq_len)

        mask_sum_before = int(text_mask[0].sum().item())
        mask_sum_after = int(new_mask[0].sum().item())

        self._log_text_anchor_expansion(
            mode, t, txt_seq_len, img_seq_len, real_end, anchor_start, anchor_end,
            alpha, mask_sum_before, mask_sum_after, real_norm_after_dual,
            _mean_norm(tiled), _mean_norm(txt[:, anchor_end:]),
        )
        self._maybe_write_anchor_token_inspection(
            mode, real_end, anchor_start, repeat, real_len,
            mask_sum_before, mask_sum_after,
        )
        return txt, cu_seqlens_q_single

    def _log_text_anchor_expansion(
        self, mode, t, txt_seq_len, img_seq_len, real_end, anchor_start, anchor_end,
        alpha, mask_sum_before, mask_sum_after, real_norm, anchor_norm, pad_norm,
    ):
        if not self._dbg_is_rank0():
            return
        try:
            timestep = float(t.flatten()[0].item())
        except Exception:
            timestep = None
        record = {
            "debug_stage": "text_anchor_expansion",
            "mode": mode,
            "prompt": os.environ.get("HUNYUAN_TEXT_ANCHOR_PROMPT"),
            "timestep": timestep,
            "txt_seq_len": int(txt_seq_len),
            "img_seq_len": int(img_seq_len),
            "real_token_range": [0, int(real_end)],
            "anchor_token_range": [int(anchor_start), int(anchor_end)],
            "pad_token_range": [int(anchor_end), int(txt_seq_len)],
            "text_mask_sum_before": int(mask_sum_before),
            "text_mask_sum_after": int(mask_sum_after),
            "anchor_alpha": float(alpha),
            "real_norm_mean_after_dual": real_norm,
            "anchor_norm_mean": anchor_norm,
            "pad_norm_mean_after_expansion": pad_norm,
        }
        path = os.environ.get(
            "HUNYUAN_TEXT_ANCHOR_DEBUG_PATH", "text_anchor_expansion_debug.jsonl"
        )
        debug_dir = os.path.dirname(path)
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _maybe_write_anchor_token_inspection(
        self, mode, real_end, anchor_start, repeat, real_len,
        mask_sum_before, mask_sum_after,
    ):
        if getattr(self, "_dbg_anchor_inspect_written", False):
            return
        if not self._dbg_is_rank0():
            return

        # Optional decoded token strings (e.g. from the .tokens.json sidecar).
        token_texts = None
        tj = os.environ.get("HUNYUAN_TEXT_ANCHOR_TOKENS_JSON", "")
        if tj and os.path.exists(tj):
            try:
                with open(tj) as f:
                    token_texts = json.load(f).get("token_texts")
            except Exception:
                token_texts = None

        def tok(i):
            if token_texts is not None and 0 <= i < len(token_texts):
                return token_texts[i]
            return None

        real_tokens = [{"index": i, "token": tok(i)} for i in range(real_end)]
        anchor_tokens = []
        for r in range(repeat):
            for j in range(real_len):
                idx = anchor_start + r * real_len + j
                anchor_tokens.append(
                    {"index": idx, "copied_from": j, "token": tok(j)}
                )

        record = {
            "prompt": os.environ.get("HUNYUAN_TEXT_ANCHOR_PROMPT"),
            "mode": mode,
            "real_tokens": real_tokens,
            "anchor_tokens": anchor_tokens,
            "text_mask_sum_before": int(mask_sum_before),
            "text_mask_sum_after": int(mask_sum_after),
        }

        path = os.environ.get(
            "HUNYUAN_TEXT_ANCHOR_DEBUG_PATH", "text_anchor_expansion_debug.jsonl"
        )
        stem = path[:-6] if path.endswith(".jsonl") else path
        out_path = f"{stem}.token_inspection.json"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        self._dbg_anchor_inspect_written = True

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,  # Should be in range(0, 1000).
        text_states: torch.Tensor = None,
        text_mask: torch.Tensor = None,  # Now we don't use it.
        text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        freqs_cos: Optional[torch.Tensor] = None,
        freqs_sin: Optional[torch.Tensor] = None,
        guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        out = {}
        img = x
        txt = text_states
        _, _, ot, oh, ow = x.shape
        tt, th, tw = (
            ot // self.patch_size[0],
            oh // self.patch_size[1],
            ow // self.patch_size[2],
        )

        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img)
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t, text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}"
            )

        # --- [debug-only] text-token norm logging (env-gated, no-op by default) ---
        # Enabled with HUNYUAN_DEBUG_TEXT_NORM=1. Snapshot the text tokens just
        # before the dual-stream blocks; they are compared against the post-dual
        # tokens right before the single-stream merge. Easy to remove: delete this
        # block, the matching `_log_text_token_norm` call below, and the method.
        _debug_text_norm = os.environ.get("HUNYUAN_DEBUG_TEXT_NORM", "0") == "1"
        # The semantic anchor expansion (below) may also need the pre-dual text.
        _anchor_mode = os.environ.get("HUNYUAN_TEXT_ANCHOR_EXPANSION_MODE", "none")
        _need_before_dual = _debug_text_norm or _anchor_mode in (
            "duplicate_before_dual",
            "duplicate_before_dual_normmatch",
        )
        txt_before_dual = txt if _need_before_dual else None

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        # Compute cu_squlens and max_seqlen for flash attention
        cu_seqlens_q = get_cu_seqlens(text_mask, img_seq_len)
        cu_seqlens_kv = cu_seqlens_q
        max_seqlen_q = img_seq_len + txt_seq_len
        max_seqlen_kv = max_seqlen_q

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        for _, block in enumerate(self.double_blocks):
            double_block_args = [
                img,
                txt,
                vec,
                cu_seqlens_q,
                cu_seqlens_kv,
                max_seqlen_q,
                max_seqlen_kv,
                freqs_cis,
            ]

            img, txt = block(*double_block_args)

        # --- [debug-only] log per-token text norms before the single-stream merge ---
        if _debug_text_norm:
            self._log_text_token_norm(txt_before_dual, txt, t)

        # --- [debug-only] text-token boundary ablation (env-gated, no-op by default) ---
        # Zero out real / padding / all text tokens right after the dual-stream
        # blocks (before the single-stream merge) to test whether padding
        # positions act as memory slots. Controlled by
        # HUNYUAN_TEXT_TOKEN_ABLATION_MODE; does not touch text before dual-stream.
        _ablation_mode = os.environ.get("HUNYUAN_TEXT_TOKEN_ABLATION_MODE", "none")
        if _ablation_mode != "none":
            txt = self._apply_text_token_ablation(txt, _ablation_mode, t, img_seq_len)

        # --- [debug-only] semantic anchor token expansion (env-gated, no-op default) ---
        # Copy clean before-dual text into unused padding positions just after the
        # real tokens and make them visible to single-stream by recomputing
        # cu_seqlens from an updated text mask. Controlled by
        # HUNYUAN_TEXT_ANCHOR_EXPANSION_MODE. Single-stream uses the recomputed
        # cu_seqlens; double-stream above is untouched.
        cu_seqlens_q_single = cu_seqlens_q
        cu_seqlens_kv_single = cu_seqlens_kv
        if _anchor_mode != "none":
            txt, cu_seqlens_q_single = self._apply_text_anchor_expansion(
                txt, txt_before_dual, text_mask, _anchor_mode, t,
                img_seq_len, txt_seq_len, cu_seqlens_q,
            )
            cu_seqlens_kv_single = cu_seqlens_q_single

        # Merge txt and img to pass through single stream blocks.
        x = torch.cat((img, txt), 1)
        if len(self.single_blocks) > 0:
            for _, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    cu_seqlens_q_single,
                    cu_seqlens_kv_single,
                    max_seqlen_q,
                    max_seqlen_kv,
                    (freqs_cos, freqs_sin),
                ]

                x = block(*single_block_args)

        img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img, vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        if return_dict:
            out["x"] = img
            return out
        return img

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double": sum(
                [
                    sum(p.numel() for p in block.img_attn_qkv.parameters())
                    + sum(p.numel() for p in block.img_attn_proj.parameters())
                    + sum(p.numel() for p in block.img_mlp.parameters())
                    + sum(p.numel() for p in block.txt_attn_qkv.parameters())
                    + sum(p.numel() for p in block.txt_attn_proj.parameters())
                    + sum(p.numel() for p in block.txt_mlp.parameters())
                    for block in self.double_blocks
                ]
            ),
            "single": sum(
                [
                    sum(p.numel() for p in block.linear1.parameters())
                    + sum(p.numel() for p in block.linear2.parameters())
                    for block in self.single_blocks
                ]
            ),
            "total": sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts


#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################

HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
}
