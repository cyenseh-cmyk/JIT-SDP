# Pre-disconnect verification for ApacheJIT

**SAFE TO DISCONNECT — All critical deliverables and backups are verified.** This verdict covers the specified scientific analyses, model evaluation and reconstruction, and rerunning the experiment from the preserved input records. Historical optimizer continuation has a separate MISSING status below.

## 1. Experiment completeness

| Final run and scope | Seed 11 | Seed 23 | Seed 37 | Validation-selected thresholds in seed order |
|---|---|---|---|---|
| v5 DeepJIT | READY | READY | READY | 0.59, 0.45, 0.61 |
| v6 CC2Vec | READY | READY | READY | 0.50, 0.50, 0.50 |
| v6 JITLine ApacheJIT adaptation | READY | READY | READY | 0.49, 0.56, 0.56 |

Each model/seed is a pooled fit evaluated on all 15 projects. The 135 project/model/seed evaluations are individually enumerated with counts, positives and recomputed metrics in [experiment-matrix.csv](experiment-matrix.csv). Thresholds are constant within a fitted model and seed. All nine completion markers match the saved experiment identity and artifact checksums. The [provenance checks](provenance-checks.json) give the exact epoch counts and runtimes.

| Earlier attempt | Status | Evidence and interpretation |
|---|---|---|
| v2 DeepJIT seed 11 | PARTIAL — failed | The saved failure record reports a missing DeepJIT model source file. No completed fit. |
| v3/v4 DeepJIT seed 11 | PARTIAL — incomplete | Setup/adapter output exists. No completion marker or final predictions. |
| v5 CC2Vec seed 11 | PARTIAL — incomplete | An intermediate representation checkpoint exists. No completed classifier or predictions. |
| v5 JITLine / v6 DeepJIT | NOT APPLICABLE to the final run allocation | The final selections use v6 JITLine and v5 DeepJIT. No earlier partial run contributes observations. |

## 2. Counterfactual completeness

The cells are applicable commit counts **per seed**, out of 30,111. All three seeds had identical eligibility. Every expected commit has an explicit row for each condition, including N/A. Across nine pipelines the audit inspected 1,354,995 rows. It found zero missing or duplicated condition keys. The [540-cell matrix](counterfactual-matrix.csv) gives the exact expected, applicable, effective-change, encoding-no-op, structural-N/A and missing counts for every project/model/seed/condition. Of those cells, 450 are READY and 90 are NOT APPLICABLE.

| Model | T1 reverse diff markers | T2 swap added/deleted blocks | T3 insert misleading markers | T4 mask message |
|---|---:|---:|---:|---:|
| DeepJIT | READY 30,111 | READY 24,494 | READY 30,111 | READY 30,111 |
| CC2Vec | READY 30,111 | READY 24,494 | READY 30,111 | NOT APPLICABLE 0 |
| JITLine | READY 21,717 | READY 17,510 | READY 30,111 | NOT APPLICABLE 0 |

DeepJIT T1 has 30,111 genuine encoding no-ops per seed because the input adapter omits line polarity. T2 requires a two-sided hunk. JITLine additionally requires changed-line totals to reconcile with the metric scope. T3 inserts the configured synthetic marker prefix. T4 empties the message for DeepJIT. CC2Vec uses messages as training targets but has no inference message channel. JITLine has no message channel either.

Independent checks used the actual test records to infer applicability and checked stored reasons, finite scores, frozen-threshold decisions, null transformed fields for N/A rows, hashes, and T1/T2 encoding equivalence flags. Frozen-model replay checked all original scores and sampled transformed inference across all 15 projects for each seed. Full transformed-score replay was not performed. The [coverage evidence](coverage-check.json) separates this exhaustive row audit from the sampled inference verification.

## 3. Artifact inventory and storage

The [218-file inventory](artifact-inventory.csv) lists each original artifact's exact container path, absolute local archive path, member name, byte size and SHA-256. All 218 have status READY for content integrity. Local supplemental files and Git histories have a separate checksum inventory. The following paths are relative to the original project root `/workspace/JIT-SDP`.

| Deliverable | Status | Artifact or verification |
|---|---|---|
| DeepJIT checkpoints and fitted preprocessing | READY | `runs/experiment_v5/deepjit/seed_{11,23,37}/classifier.pt` and `adapter.json` |
| CC2Vec representation and classifier | READY | `runs/experiment_v6/cc2vec/seed_{11,23,37}/encoder.pt`, `classifier.pt`, `adapter.json` |
| JITLine forest, vectorizer and scaler | READY | `runs/experiment_v6/jitline_apachejit/seed_{11,23,37}/forest.joblib`. Each forest loaded with 300 trees, 50,000 token features and 12 scaled metrics. |
| Per-commit predictions and labels | READY | `predictions.csv` under each of the nine seed directories. Original labels match the published ApacheJIT CSVs. Labels for genuinely new counterfactual outcomes are unavailable by study design. |
| Thresholds, preprocessing limits and evaluation | READY | Each seed's `threshold.json`, `encoding_audit.json`, `runtime.json`, `complete.json`. The [replay evidence](checkpoint-and-threshold-replay.json) gives the score and decision discrepancies. |
| Training/evaluation logs | PARTIAL | Structured `history.json` for every seed, including both neural stages where applicable, plus final metrics, predictions and fit runtimes. JITLine records resampling/fitted row count and its separate candidate log. Complete terminal transcripts, per-batch traces, and complete end-to-end timing including exploratory work were not found among the experiment artifacts. |
| Optimizer, scheduler, RNG and exact-step continuation | MISSING for neural optimizer/RNG continuation; NOT APPLICABLE to forest Adam state | PyTorch files contain finite model-state tensors. The saver writes model state dictionaries; the Adam state, epoch continuation state, and random-generator positions are not persisted. No scheduler was used by this loop. |
| Splits and preprocessing implementation | READY | `data/manifests/{train,validation,test}.jsonl`, `split.json`, `content_overlap.json`, `src/jit_sdp/features.py`, per-seed adapters. Original archive, enrichment database and cleaned dataset are also backed up. |
| Source, configs and environment | READY for inspection/restoration | Frozen `runs/experiment_v{5,6}/experiment.json` and `hardware.json`, `configs/default.json`, `configs/experiment_v6.json`, source archive, [live environment](live-environment.json), [Git revisions and bundles](verified-git-bundles.json), and `v5-source-overlay/src/jit_sdp/{config,neural}.py`. |
| Baseline/stress results and provenance | READY | `reports/experiment_v5_v6/{comparison.json,comparison.md,sources.json}` inside the runs archive. Three standalone local copies match those archived bytes. The v6-only report contains two methods. |
| Original validation per-commit probability export | MISSING as a standalone historical table | The saved validation records and frozen pipelines reproduce the selected thresholds and validation metrics. New validation score tables can be generated by inference. |
| Full Apache Git mirrors and installed virtual environment | PARTIAL for offline raw-history reproduction | The host mirrors and virtual environment are outside the original backup. The published dataset, raw enrichment cache, prepared splits, baseline implementation sources, package versions and mirror HEAD IDs are preserved. |

## 4. Integrity exceptions and missing deliverables

No final checkpoint, prediction table, split or critical archive failed integrity verification. A supplemental Git transfer was interrupted. Its retained `git-bundles/JIT-SDP.bundle.partial` is **CORRUPT/incomplete** and excluded from the usable backup. All four replacements in `verified-bundles/` passed SHA-256 verification, Git bundle verification, and an actual fetch of the expected commit.

No recorded outputs permit exact continuation of the historical Adam trajectory. A new fit could create new optimizer states, but could not establish an authentic historical state. Checkpoint evaluation and requested statistical tests require no retraining. Re-creating exact historical console output is also unavailable. Compute-cost reporting should describe the scope of the saved runtimes.

## 5. Statistical analysis readiness

| Question | Status | Verified basis and outputs |
|---|---|---|
| RQ1 baseline performance | READY | Recomputed MCC, binary F1 and ROC–AUC for each seed/project and pooled sample from probabilities and hard labels. The [RQ1 supplement](rq1-fixed-project-bootstrap.csv) supplies three-seed means, sample SDs, retained/undefined counts, and 95% intervals from 1,000 paired draws. [Matched baseline contrasts](rq1-paired-baseline-contrasts.csv) use the same commits at each draw for all methods and seeds. |
| RQ2 perturbation sensitivity | READY on applicable cohorts | [Pooled tests](rq2-fixed-project-bootstrap-and-permutation.csv) and [project intervals](rq2-by-project-intervals.csv) use the exact paired keys and report original MCC minus transformed MCC. Pooled tests use 9,999 within-commit random swaps shared across seeds, a two-sided plus-one p-value, and Holm across the ten applicable model/condition tests. Equal label-and-prediction patterns are grouped into binomial swap counts, which preserve this permutation distribution. |
| RQ3 family comparison | READY for T1–T3; NOT APPLICABLE for T4 | [Matched project effects](rq3-matched-project-effects.csv) are averaged over all three seeds per method before inference. The [Friedman supplement](rq3-exact-friedman.csv) reports both asymptotic and exact within-project model-label permutation p-values. The [Wilcoxon supplement](rq3-exact-wilcoxon.csv) records zero differences, ties, project counts, sign-enumerated p-values and Holm across the six performed follow-ups. T2 follow-ups are gated by its omnibus result. |

**Important correction to interpretation.** The original report's pooled bootstrap resamples projects as well as commits. The brief holds observed project sizes fixed and resamples commits within each project. Both estimands are explicitly identifiable. The new supplement implements the brief's conditional analysis with analysis seed 2026. For JITLine minus DeepJIT, mean MCC differs by +0.01423. The original project-resampling interval was approximately [−0.105, +0.157]; the fixed-project paired interval is [+0.00416, +0.02330]. Use the interval corresponding to the scientific claim. The latter supports an advantage on these observed projects and does not establish superiority on new projects.

T1, T2 and T3 respectively have 21,717, 17,510 and 30,111 matched commits across all three families. Each has 14 evaluable project blocks. Apache Hadoop has 5,421 test commits and no positives. Its model outputs are complete; MCC and ROC–AUC are undefined for that project. The supplements preserve undefined and unstable bootstrap cohorts explicitly. T4 has only one eligible method. RQ3 exact omnibus p-values for T1, T2 and T3 are 0.00042665, 0.21971 and 0.00094410. Each exact distribution covers 6^14 within-project label permutations, accounting for ties. This is a post-run audit supplement. The artifacts do not prove that every inferential choice was prospectively registered.

Stress results concern predictions under altered representations, evaluated against original labels. The source split records four exact-content overlaps across validation and test with distinct commit IDs. Report those overlaps, project/commit dependence, label availability over time, adaptation of the CC2Vec classifier and JITLine features, and input truncation. No conclusion about executable counterfactual correctness follows from transformed-label MCC.

## 6. Reproducibility

- **READY — cohort and fit inputs.** The partition content SHA-256 digests and ordered-time boundaries were checked. Train/validation/test contain 35,860, 8,965 and 30,111 commits. Key sets are disjoint. The test IDs and labels agree with the original dataset; all 74,936 included labels were checked against published membership. The 74,945 cached candidate records include nine exclusions. SQLite returned `ok` for its read-only quick integrity check.
- **READY — preprocessing.** Recomputed train-only neural token mappings and the JITLine vectorizer and scaler matched the saved instances. JITLine metric order is `la, ld, nf, nd, ns, ent, ndev, age, nuc, aexp, arexp, asexp`. Validation and test consumed the saved transforms.
- **READY — checkpoint inference.** All nine pipelines loaded using the recorded project configuration on the live runtime. The restored local artifacts have identical SHA-256s. Full baseline replay covered 270,999 scores and reproduced every decision, with maximum absolute probability error 1.12e-16. For each pipeline one test commit per project was replayed across all five conditions, including explicit N/A. All sampled input hashes, status codes and decisions matched. On all 8,965 validation commits per pipeline, the frozen threshold was independently reselected over the configured grid; validation metric differences were below 8e-8.
- **READY — code provenance.** All package-source hashes and upstream Python files were checked against the run records. The original backup held the v6 source. Two v5 Python files were reconstructed and accepted only after their bytes matched the v5 SHA-256s. Four source Git histories were independently fetched locally from the saved bundles. For a v5 reconstruction, overlay the two verified v5 files on the restored package tree. Keep v6's batch-lines configuration for CC2Vec.
- **PARTIAL — exact environment reconstruction.** The repository lock differs from the recorded runtime for NumPy and SciPy. The audit captured installed distribution versions, Python, OS, GPU, driver, CUDA/PyTorch build details, and the Git revision identifiers. Restore into a compatible NVIDIA/Linux environment, rebase absolute data and baseline paths from the original configuration, and use the recorded package versions. The environment snapshot is an inventory, not an immutable machine image.
- **NOT VERIFIED — bit-for-bit fresh training or a clean-host end-to-end rebuild.** The checks exercised inference, preprocessing and statistics without fitting a prediction model. Historical optimizer continuation is MISSING. The seed-averaging convention uses all three fixed seeds, with sample SD across seeds; projects are the independent blocks assumed by the family comparison.

## 7. Backup verification and persistence

The three local gzip/tar archives contained 218 unique regular files totalling 11,747,894,075 source bytes. Their archive-level SHA-256s and every member's byte count and SHA-256 were independently checked. A subsequent read of all 218 live files found zero differences. The local Windows Projects folder is independent of the Vast container. These checks establish a usable backup on one computer, not an independent second-drive or cloud copy.

The container's capability manifest reports `workspace_is_volume=false`; `/workspace` resolves to the container overlay. Treat the container as temporary. The audit outputs, v5 overlay and usable Git bundles are in this local report directory. Temporary extraction paths are convenience copies, not the authoritative backup.

## 8. Prioritized actions

1. **Before disconnecting:** no critical publication, frozen-model or saved-input reproduction artifact remains to copy. If an additional objective requires all fifteen original Git mirrors for fully offline history acquisition, or an immutable runtime image, preserve those separately before destroying the instance. The stated scoped verdict does not certify those optional backups.
2. **Before writing claims:** use the fixed-project tables for the brief's primary analysis. Name the old pooled results as a different project-resampling sensitivity analysis. State the one-class project, N/A message channels, truncation, coverage, four overlaps, corrected comparison families and runtime scope.
3. **For long-term retention:** copy the three original archives, original manifest, and this entire audit directory to a second durable location. Use the [file inventory](artifact-inventory.csv), original manifest and local audit checksums to verify the copy. A clean-host inference rehearsal is advisable before distributing the replication package.
4. **For a future resumable training study:** arrange optimizer, epoch, RNG and step-state checkpoints prospectively. No recovery procedure can extract an unsaved historical optimizer state from the final model parameters.
