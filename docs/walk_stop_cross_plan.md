# Historical walk-stop-cross implementation plan

Current p3 design: `superpowers/specs/2026-10-02-fixed-stick-moving-design.md`.
Current training and deployment instructions: `walk_stop_cross_hpc_zh.md`.

Goal: fine-tune the existing 49-observation policy on v3.2, spawning the bar only after a continuous stable stop.

- [x] Add tested vectorized stop gate: double support, speed < 0.05 m/s, yaw rate < 0.10 rad/s, continuous 0.5 s; fail after 3 s.
- [x] Integrate gate into shared crossing state and command priority; reset only selected environments and spawn from current feet geometry.
- [x] Gate stepping rewards while stopped; add stop stability/completion rewards and timeout termination; require actual crossing and landing for success.
- [x] Keep Phase 5 mechanics and observation/action interfaces; give fine-tuning an independent schedule identity. Use v3.2, 80% bar episodes, 8 s episodes, fixed 5e-5 LR, clip 0.1 and entropy 5e-4.
- [x] Add explicit warm-start versus resume training entry point, independent logs, actual optimizer LR checks, metrics and checkpoint validation.
- [x] Run CPU logic and regression tests, syntax checks and inspect the diff. Run simulator smoke training only if Isaac Lab is available; document any missing runtime.

Accepted design: stop before spawning the bar; load model_33999.pt for the first fine-tune, reset optimizer and fine-tune counters. Full state is restored only for continuation of this fine-tune. Preserve existing user changes and original project.

HPC execution is pending. Local simulator smoke runs were limited to 2 and 25 updates per the user's instruction.

