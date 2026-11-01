# Recorded experiment configurations

`primary_run_manifest.csv` lists the current 22 primary runs and links each to its log, argument dictionary, and per-epoch data. Every included run has configured poisoning 0.01. GS seed 45 is the added RTX 3090/workers-0 run; the other primary runs use the recorded RTX 4090/workers-8 setting.

`isolated_control_manifest.csv` lists nine RNG-isolated runs: Fixed, Free, and PSNR-matched Fixed with seeds 41/42/43. The three Free runs are shared between the two comparisons. `run_reuse.json` records other reused runs.

`logged_args/` and `log_configuration_index.csv` preserve arguments extracted from the original logs. Recorded dataset and output paths describe the original machine; replace these paths for a new run while retaining the scientific settings. Original numerical logs remain unchanged.

The executable entries are `backdoorbench.attack.wanet` (C1) and `backdoorbench.attack.wanet_rng_control` (C2). C2 saves and restores evaluation RNG states. The active files have revised comments and documentation; their executable statements retain the corresponding received implementation.

`original_yaml/` retains the supplied defaults. Explicitly override the default poisoning rate of 0.1 for the paper's configured 1% comparisons. Check the relevant recorded arguments for model, seed, target, workers, seed resolution, optimizer settings, and the C1/C2 entry.
