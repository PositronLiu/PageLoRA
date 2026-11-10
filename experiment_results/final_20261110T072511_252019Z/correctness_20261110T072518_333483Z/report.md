# Correctness report

Strict pass: True

| Candidate (reference: HF unless noted) | Case | Exact tokens | First divergence | Logits allclose | Cache reused | Graph replayed |
|---|---|---|---|---|---|---|
| flashinfer_eager | prefix_31 | True | None | True | True | n/a |
| flashinfer_eager | prefix_32 | True | None | True | True | n/a |
| flashinfer_eager | prefix_33 | True | None | True | True | n/a |
| flashinfer_eager | prefix_129 | True | None | True | True | n/a |
| flashinfer_graph | prefix_31 | True | None | True | True | True |
| flashinfer_graph | prefix_32 | True | None | True | True | True |
| flashinfer_graph | prefix_33 | True | None | True | True | True |
| flashinfer_graph | prefix_129 | True | None | True | True | True |
| vllm | prefix_31 | True | None | n/a | n/a | n/a |
| vllm | prefix_32 | True | None | n/a | n/a | n/a |
| vllm | prefix_33 | True | None | n/a | n/a | n/a |
| vllm | prefix_129 | True | None | n/a | n/a | n/a |
| graph_vs_eager | prefix_31 | True | None | n/a | n/a | n/a |
| graph_vs_eager | prefix_32 | True | None | n/a | n/a | n/a |
| graph_vs_eager | prefix_33 | True | None | n/a | n/a | n/a |
| graph_vs_eager | prefix_129 | True | None | n/a | n/a | n/a |

## Missing evidence

None

## Scope

- vLLM compares greedy token IDs only, not full logits.
- First-step logits cover prefill; later steps compare sequences, not teacher-forced logits.
- A mismatch requires inspecting first divergence and floating-point margins; it alone does not prove a bug.
- These cases cover repeated prompts and page boundaries, not concurrent serving or OOM recovery.
- Graph validation requires a nonzero count of successful runner replays.
- Logits tolerances default to atol=0.1, rtol=0.01; these are screening settings, not universal error bounds.
