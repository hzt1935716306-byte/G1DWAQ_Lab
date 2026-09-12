# Parallel-environment performance freeze

Benchmark date: 2026-09-12. Device: NVIDIA GeForce RTX 4090 (24 GB).
Environment: Conda `g1`, Isaac Sim 5.1.0.0, Isaac Lab 0.54.2. Model: M5,
checkpoint SHA-256 `9bde392d2a91ef7e20bfd0b6e84625e2663caf205577f02f0b720f1830a879c1`.
Each candidate used 10 warm-up and 50 measured 50 Hz policy steps with the
same native environment, checkpoint, simulator process isolation, and physics
profile (`e8ec92183ec41d1bf1b1d56add684954f192d1ead4fd4dc98dd6e3f80526a6cf`).

| Environments | Wall time (s) | Env steps/s | Valid simulated s / wall s |
|---:|---:|---:|---:|
| 64 | 1.7756 | 1,802.20 | 36.04 |
| 128 | 2.4911 | 2,569.10 | 51.38 |
| 256 | 3.4986 | 3,658.64 | 73.17 |
| 512 | 4.7076 | 5,438.05 | 108.76 |
| 1024 | 7.1636 | 7,147.22 | 142.94 |
| 2048 | 12.0376 | 8,506.67 | 170.13 |

The frozen formal value is **2048 environments**, which maximized the declared
selection metric. Small technical smoke and screening runs may pass a smaller
explicit `--num-envs` to avoid padding a tiny trial set; this does not alter the
formal default. The recorded `peak_cuda_memory_MiB` field is PyTorch allocator
memory only and is not used as the selection metric because PhysX/Vulkan memory
is managed outside that allocator.
