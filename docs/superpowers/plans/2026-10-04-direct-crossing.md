# Stage 2 direct-crossing curriculum, 2026-10-04

User-approved levels (walk/gap cm): 10/8, 8/7, 6/6, 4/5, 2/4, 0/3.
Keep physical bar collision and all motor/reward constraints. Default train and play physics stage is 2.

1. Add CPU tests for strict >70%, 200 minimum episodes, cohort isolation, frozen replay, save/restore and zero-step direct LEAD.
2. Implement a simulator-independent curriculum with per-episode level tags and window clearing on promotion.
3. Apply geometry/commands only at reset; terminal zero-step starts LEAD, right foot first, without WALK bonus.
4. Integrate terminal episode reporting, checkpoint metadata, training resume and frozen checkpoint-level replay.
5. Make default train/play stage 2, retain old weights for warm starts without requiring Stage 1 qualification.
6. Run CPU regression and native Isaac Stage 2 training/replay smoke if available; document limitations and commands.
