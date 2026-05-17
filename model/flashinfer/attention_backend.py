"""FlashInfer backend wrapper.

This module centralizes FlashInfer wrapper API calls so the rest of the runtime
can stay independent from FlashInfer version details.
"""

from __future__ import annotations

import flashinfer
import torch


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
        size = self.workspace_mb * 1024 * 1024
        self.workspace_buffer = torch.empty(size, dtype=torch.uint8, device=self.device)
        self._decode_wrapper = self._make_wrapper(flashinfer.BatchDecodeWithPagedKVCacheWrapper)
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

        plan_kwargs = {
            "indptr": paged_kv_indptr,
            "indices": paged_kv_indices,
            "last_page_len": paged_kv_last_page_len,
            "num_qo_heads": self.num_qo_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
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
        output = self._call_run(self._decode_wrapper, q_nhd, paged_kv_cache_layer)
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
            output = self._call_run(self._prefill_wrapper, q_nhd, paged_kv_cache_layer)
            if isinstance(output, tuple):
                output = output[0]
            return output.reshape(bsz, seqlen, num_heads, head_dim).transpose(1, 2).contiguous()

        output = self._call_run(self._prefill_wrapper, q, paged_kv_cache_layer)
        if isinstance(output, tuple):
            output = output[0]
        return output

    def _make_wrapper(self, wrapper_cls):
        try:
            return wrapper_cls(self.workspace_buffer, kv_layout="NHD")
        except TypeError:
            return wrapper_cls(self.workspace_buffer, "NHD")

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

        fallback_kwargs = {"pos_encoding_mode": "NONE"}
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

        minimal_kwargs = {"pos_encoding_mode": "NONE"}
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
        )

    @staticmethod
    def _call_run(wrapper, q: torch.Tensor, paged_kv_cache_layer: torch.Tensor):
        try:
            return wrapper.run(q, paged_kv_cache_layer)
        except TypeError:
            return wrapper.run(q, paged_kv_cache_layer, return_lse=False)
