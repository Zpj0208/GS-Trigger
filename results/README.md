# Experimental records

`raw_archives/` contains original numerical records, logs, configurations, and RNG traces. Internal handoff notes and acceptance reports are not included. `data_manifests/source_files.csv` records the distributed source-file hashes.

`primary/epoch_metrics/` contains data extracted from all 100 epochs of the current 22 primary logs. `validated/` contains source-based result tables. For the rate-resolution grid, use `grid16_corrected_summary.csv` and `grid16_epoch_metrics_from_original_logs.csv`; historical mislabeled wrappers remain archival records.

`recomputed/` is created by the result-check scripts. Means and sample SD are computed across independent training runs. For final-ten-epoch summaries, each run contributes one window mean before across-run SD is calculated. Epochs, gradient events, image pairs, and calibration budgets are not independent training repeats.

The original Free seed-42 strength-control run has 90.00% final ASR. The 97.07% mixed-gradient run is a separate diagnostic. Dataset extensions have one training run per condition. Defense data describe recorded checkpoint cases; effective repairs and unfavorable outcomes are retained.
