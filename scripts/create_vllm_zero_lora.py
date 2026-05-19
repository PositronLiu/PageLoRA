from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from transformers import AutoConfig


TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj")


def _dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def _module_shapes(config, target_modules: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    hidden_size = int(config.hidden_size)
    head_dim = int(getattr(config, "head_dim", hidden_size // int(config.num_attention_heads)))
    kv_hidden_size = int(config.num_key_value_heads) * head_dim
    supported = {
        "q_proj": (hidden_size, hidden_size),
        "k_proj": (hidden_size, kv_hidden_size),
        "v_proj": (hidden_size, kv_hidden_size),
        "o_proj": (hidden_size, hidden_size),
    }
    unknown = sorted(set(target_modules) - set(supported))
    if unknown:
        raise ValueError(f"unsupported target modules for zero LoRA generation: {unknown}")
    return {name: supported[name] for name in target_modules}


def build_zero_lora_tensors(
    *,
    num_layers: int,
    module_shapes: dict[str, tuple[int, int]],
    rank: int,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    for layer_idx in range(num_layers):
        for module_name, (in_features, out_features) in module_shapes.items():
            prefix = f"base_model.model.model.layers.{layer_idx}.self_attn.{module_name}"
            tensors[f"{prefix}.lora_A.weight"] = torch.zeros(rank, in_features, dtype=dtype)
            tensors[f"{prefix}.lora_B.weight"] = torch.zeros(out_features, rank, dtype=dtype)
    return tensors


def write_adapter_config(
    *,
    output_dir: Path,
    model_path: str,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: tuple[str, ...],
) -> None:
    config = {
        "base_model_name_or_path": model_path,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": alpha,
        "lora_dropout": dropout,
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": list(target_modules),
        "task_type": "CAUSAL_LM",
    }
    (output_dir / "adapter_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def create_zero_lora_adapter(
    *,
    model_path: str,
    output_dir: Path,
    rank: int,
    alpha: int,
    dropout: float,
    target_modules: tuple[str, ...] = TARGET_MODULES,
    dtype: torch.dtype = torch.float16,
) -> None:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    module_shapes = _module_shapes(config, target_modules)
    tensors = build_zero_lora_tensors(
        num_layers=int(config.num_hidden_layers),
        module_shapes=module_shapes,
        rank=rank,
        dtype=dtype,
    )
    save_file(tensors, str(output_dir / "adapter_model.safetensors"), metadata={"format": "pt"})
    write_adapter_config(
        output_dir=output_dir,
        model_path=model_path,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a vLLM-compatible zero LoRA adapter for benchmark smoke tests.")
    parser.add_argument("--model", required=True, help="Base HuggingFace model path or name.")
    parser.add_argument("--output", required=True, help="Output adapter directory.")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--dropout", type=float, default=0.0, help="LoRA dropout in adapter_config.json.")
    parser.add_argument(
        "--target-modules",
        default=",".join(TARGET_MODULES),
        help="Comma-separated target module names. Default: q_proj,k_proj,v_proj,o_proj.",
    )
    parser.add_argument("--dtype", default="float16", help="Adapter tensor dtype: float16, bfloat16, or float32.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    target_modules = tuple(part.strip() for part in args.target_modules.split(",") if part.strip())
    create_zero_lora_adapter(
        model_path=args.model,
        output_dir=Path(args.output),
        rank=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=target_modules,
        dtype=_dtype_from_name(args.dtype),
    )
    print(f"wrote vLLM zero LoRA adapter to {args.output}")


if __name__ == "__main__":
    main()
