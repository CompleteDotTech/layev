# Bounded long-context training memory evidence

This change targets Layev issue #4. During long CUDA training it keeps each
grouped-query key/value head as a read-only expanded view for fused attention,
and checkpoints 1,024-row MLP slices inside the existing layer checkpoint.
Short paths, inference arithmetic, the batch policy, the 32,768/65,536 limits,
and the declared numerical tolerances are unchanged. Unsupported fused
attention kernels still fail rather than falling back to dense math attention.

On an RTX 3060 with PyTorch 2.10.0+cu128 and a 70% per-process GPU cap, the
source-tree candidate completed one optimizer step on a generated 19-question
training example with 32,768 maximum branch and 65,536 aggregate useful
tokens. It took 90.234 seconds, used one prefix pass and ten branch passes
(nine paired batches and one singleton), and recorded 8,568,420,864 bytes
peak PyTorch allocation. The checkpoint SHA-256 is
`0fb62b215e9925644dcf899467e3ec1160845ce38e5946743388c1ae51b229d0`;
the telemetry SHA-256 is
`aff5733ee2e4111d4a0cbce9c89677ac50b0f7911d5d41ac14e51d19e478d362`.
This first source-tree run preceded the signed commit, so its source provenance
correctly records a dirty checkout. A separate clean signed-head replay of the
same source change completed two optimizer steps with a parent-linked resume,
and the step-2 checkpoint reloaded for finite exact-boundary inference. The
private execution ledger preserves the exact replay hashes and provenance.

The dedicated CUDA tests compare the new long grouped-attention path with
materialized heads, including input and parameter gradients, and the chunked
MLP with the full-row FP32 path at unchanged 1e-5 output and 2e-5 gradient
tolerances. Both passed locally. The CPU project suite passed 819 tests with
18 skips, including the two CUDA-only tests. Compilation and critical Ruff
checks passed.

Earlier private monkeypatch diagnostics completed two synthetic steps with
durable resume, but their code is not represented by a Layev source commit.
The signed-source replay is still generated smoke data. It does not establish
calibration, exact/overflow service responses, representative quality, Jev
comparison, or the full issue #4 artifact chain. Preserve those gates and the
earlier failed attempts.
