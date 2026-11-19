# ApacheJIT final experiment results

The final comparison uses DeepJIT from experiment v5 and CC2Vec and the 12-metric JITLine adaptation from v6. Each method was trained with seeds 11, 23 and 37 and evaluated on the same 30,111 test commits.

| Method | Mean MCC | Mean binary F1 | Mean ROC-AUC |
|---|---:|---:|---:|
| DeepJIT | 0.3453 | 0.4799 | 0.7836 |
| CC2Vec | 0.1229 | 0.3416 | 0.5944 |
| JITLine | 0.3596 | 0.4920 | 0.7968 |

Use the [fixed-project bootstrap estimates](audit/rq1-fixed-project-bootstrap.csv) and [paired contrasts](audit/rq1-paired-baseline-contrasts.csv) for inference conditional on the observed projects. The [original report](comparison.json) also varies the mix of projects. The [audit](audit/AUDIT.md) documents the two estimands, model replay checks, stress-test applicability and recovery limits.

The [run records](run-records/) contain frozen configurations, validation thresholds, training histories, timings and artifact checksums. The [source manifest](sources.json) identifies each model's producing run.

## Recompute the audit tables

Extract the project, prepared-data and runs archives into a single project directory and install the analysis dependencies. Pass that directory as the first argument and a fresh output directory as the second. Run the commands from the root of this checkout.

```sh
python scripts/experiment_analysis/audit_predictions.py /path/to/restored-project /path/to/new-analysis
python scripts/experiment_analysis/supplement_statistics.py /path/to/restored-project /path/to/new-analysis
python scripts/experiment_analysis/paired_baseline_contrasts.py /path/to/restored-project /path/to/new-analysis
```

For frozen-model replay, the saved configurations reference the original Linux project root, `/workspace/JIT-SDP`. The audit records the environment and the two v5 source files needed to reconstruct its exact version.
