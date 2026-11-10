"""FlashInfer backend wrapper.

This module centralizes FlashInfer wrapper API calls so the rest of the runtime
can stay independent from FlashInfer version details.
"""

from __future__ import annotations

import math

import flashinfer
import torch

from core.utils import LOG


class FlashInferAttentionBackend:
    """Own FlashInfer workspace allocation and wrapper calls."""

    def __init__(
        self,
        *,
        device: torch.device,
        workspace_mb: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
    ) -> None:
        self.device = device
        self.workspace_mb = max(1, int(workspace_mb))
        self.num_qo_heads = int(num_qo_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.page_size = int(page_size)
        # FlashInfer 0.6.6 uses finite -5e4 mask sentinels before score
        # scaling. Qwen2's final-layer raw QK products can be more negative,
        # letting masked positions win. Binary rescaling preserves QK*scale
        # while moving unscaled products away from that sentinel.
        self.query_rescale = 16.0
        size = self.workspace_mb * 1024 * 1024
        self.workspace_buffer = torch.empty(size, dtype=torch.uint8, device=self.device)
        if self.num_kv_heads <= 0 or self.num_qo_heads % self.num_kv_heads:
            raise ValueError('Q head count must be divisible by KV head count')
        # The CUDA-core decode kernel only specializes these GQA ratios.
        # Qwen2 7B has 28 Q heads / 4 KV heads = 7; use the tensor-core
        # implementation rather than failing in plan() with Unsupported group_size.
        self.decode_use_tensor_cores = self.num_qo_heads // self.num_kv_heads not in (1, 2, 4, 8)
        if self.decode_use_tensor_cores:
            self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer, kv_layout="NHD", use_tensor_cores=True
            )
        else:
            self._decode_wrapper = self._make_wrapper(flashinfer.BatchDecodeWithPagedKVCacheWrapper)
        self._decode_graph_buffers = None
        self._prefill_wrapper = self._make_wrapper(flashinfer.BatchPrefillWithPagedKVCacheWrapper)

    def plan_decode(
        self,
        *,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        q_data_type: torch.dtype | None = None,
        kv_data_type: torch.dtype | None = None,
    ) -> None:
        """Plan one FlashInfer paged decode step.

        The plan is batch/request-level metadata only and can be reused for all
        decoder layers in the same generation step.
        """

        self._ensure_graph_safe_decode_buffers(paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len)
        plan_kwargs = {
            "indptr": paged_kv_indptr,
            "indices": paged_kv_indices,
            "last_page_len": paged_kv_last_page_len,
            "num_qo_heads": self.num_qo_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
            "sm_scale": self.query_rescale / math.sqrt(self.head_dim),
            # Q/K RoPE is applied before cache writes by the generation path.
            # The paged attention wrapper must not apply RoPE again.
            "pos_encoding_mode": "NONE",
            "q_data_type": q_data_type,
            "kv_data_type": kv_data_type,
        }
        self._call_plan(
            self._decode_wrapper,
            plan_kwargs,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )

    def run_decode(
        self,
        q: torch.Tensor,
        paged_kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        """Run one layer of FlashInfer paged decode attention."""

        q_nhd = q.squeeze(2) if q.dim() == 4 and q.size(2) == 1 else q
        output = self._call_run(self._decode_wrapper, q_nhd / self.query_rescale, paged_kv_cache_layer)
        if isinstance(output, tuple):
            output = output[0]
        return output.unsqueeze(2) if output.dim() == 3 else output

    def plan_prefill(
        self,
        *,
        qo_indptr: torch.Tensor,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        paged_kv_last_page_len: torch.Tensor,
        causal: bool = True,
        q_data_type: torch.dtype | None = None,
        kv_data_type: torch.dtype | None = None,
    ) -> None:
        """Plan FlashInfer paged prefill attention."""

        plan_kwargs = {
            "qo_indptr": qo_indptr,
            "paged_kv_indptr": paged_kv_indptr,
            "paged_kv_indices": paged_kv_indices,
            "paged_kv_last_page_len": paged_kv_last_page_len,
            "num_qo_heads": self.num_qo_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim_qk": self.head_dim,
            "page_size": self.page_size,
            "sm_scale": self.query_rescale / math.sqrt(self.head_dim),
            "causal": causal,
            # Q/K RoPE is applied before cache writes by the generation path.
            # The paged attention wrapper must not apply RoPE again.
            "pos_encoding_mode": "NONE",
            "q_data_type": q_data_type,
            "kv_data_type": kv_data_type,
        }
        self._call_plan(
            self._prefill_wrapper,
            plan_kwargs,
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
        )

    def run_prefill(
        self,
        q: torch.Tensor,
        paged_kv_cache_layer: torch.Tensor,
    ) -> torch.Tensor:
        """Run one layer of FlashInfer paged prefill attention."""

        if q.dim() == 4:
            bsz, num_heads, seqlen, head_dim = q.shape
            q_nhd = q.transpose(1, 2).reshape(bsz * seqlen, num_heads, head_dim)
            output = self._call_run(self._prefill_wrapper, q_nhd / self.query_rescale, paged_kv_cache_layer)
            if isinstance(output, tuple):
                output = output[0]
            return output.reshape(bsz, seqlen, num_heads, head_dim).transpose(1, 2).contiguous()

        output = self._call_run(self._prefill_wrapper, q / self.query_rescale, paged_kv_cache_layer)
        if isinstance(output, tuple):
            output = output[0]
        return output

    def _make_wrapper(self, wrapper_cls):
        try:
            return wrapper_cls(self.workspace_buffer, kv_layout="NHD")
        except TypeError:
            return wrapper_cls(self.workspace_buffer, "NHD")

    def _ensure_graph_safe_decode_buffers(self, indptr, indices, last_page_len):
        # Without use_cuda_graph, tensor-core plan() allocates a fresh qo_indptr
        # on every call. Replaying an older graph then dereferences freed memory.
        if self.device.type != 'cuda':
            return
        buffers = self._decode_graph_buffers
        if buffers is not None and buffers[0].numel() == indptr.numel() and buffers[1].numel() >= indices.numel():
            return
        capacity = max(16, 1 << max(0, indices.numel() - 1).bit_length())
        buffers = (torch.empty_like(indptr),
                   torch.empty(capacity, device=self.device, dtype=indices.dtype),
                   torch.empty_like(last_page_len))
        self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self.workspace_buffer, kv_layout='NHD', use_cuda_graph=True,
            use_tensor_cores=self.decode_use_tensor_cores,
            paged_kv_indptr_buffer=buffers[0], paged_kv_indices_buffer=buffers[1],
            paged_kv_last_page_len_buffer=buffers[2],
            **({'backend': 'fa2'} if self.decode_use_tensor_cores else {}),
        )
        self._decode_graph_buffers = buffers
        LOG('FlashInfer decode wrapper: '
            f'tensor_cores={self.decode_use_tensor_cores} use_cuda_graph=True '
            f'batch={last_page_len.numel()} index_capacity={capacity} '
            f'backend={"fa2" if self.decode_use_tensor_cores else "auto"}')

    def decode_graph_signature(self):
        wrapper = self._decode_wrapper
        # Host-side planning parameters are captured by value, unlike page-table
        # tensors. An altered plan must not reuse an older captured launch.
        return (id(wrapper), tuple(getattr(wrapper, '_plan_info', None) or ()))

    def _call_plan(
        self,
        wrapper,
        plan_kwargs: dict[str, object],
        *positional,
    ) -> None:
        compact_kwargs = {key: value for key, value in plan_kwargs.items() if value is not None}
        try:
            wrapper.plan(**compact_kwargs)
            return
        except TypeError:
            pass

        fallback_kwargs = {"pos_encoding_mode": "NONE", "sm_scale": plan_kwargs["sm_scale"]}
        if "causal" in plan_kwargs:
            fallback_kwargs["causal"] = plan_kwargs["causal"]
        if plan_kwargs.get("q_data_type") is not None:
            fallback_kwargs["q_data_type"] = plan_kwargs["q_data_type"]
        if plan_kwargs.get("kv_data_type") is not None:
            fallback_kwargs["kv_data_type"] = plan_kwargs["kv_data_type"]
        try:
            wrapper.plan(
                *positional,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                **fallback_kwargs,
            )
            return
        except TypeError:
            pass

        minimal_kwargs = {"pos_encoding_mode": "NONE", "sm_scale": plan_kwargs["sm_scale"]}
        if "causal" in plan_kwargs:
            minimal_kwargs["causal"] = plan_kwargs["causal"]
        try:
            wrapper.plan(
                *positional,
                self.num_qo_heads,
                self.num_kv_heads,
                self.head_dim,
                self.page_size,
                **minimal_kwargs,
            )
            return
        except TypeError:
            pass

        wrapper.plan(
            *positional,
            self.num_qo_heads,
            self.num_kv_heads,
            self.head_dim,
            self.page_size,
            sm_scale=plan_kwargs["sm_scale"],
        )

    @staticmethod
    def _call_run(wrapper, q: torch.Tensor, paged_kv_cache_layer: torch.Tensor):
        try:
            return wrapper.run(q, paged_kv_cache_layer)
        except TypeError:
            return wrapper.run(q, paged_kv_cache_layer, return_lse=False)
