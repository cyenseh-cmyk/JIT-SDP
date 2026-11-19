# ApacheJIT model comparison

A Python package for the protocol in `JIT_Defect_Prediction_Project_Brief.pdf`:
archive audit → Git enrichment → common eligible cohort → fixed temporal split →
DeepJIT / CC2Vec / JITLine (ApacheJIT adaptation) → saved predictions and comparison.

The CSVs contain **no messages or patches**. Real training requires Apache Git history.
The archive and cloned baseline sources are preserved. No full Apache training run has
been performed merely by creating this package.

## Install

After cloning or pulling this repository, fetch the pinned baseline checkouts from the project root:

```powershell
git submodule update --init --recursive
```

Use Python 3.11 or newer and Git. From this project directory in PowerShell:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e .
.venv/Scripts/python.exe -m jit_sdp --config configs/default.json doctor
```

`requirements-lock.txt` records the versions actually used to verify this package.
For another machine, install that file before `pip install --no-deps -e .`.
The default installation uses the available PyTorch build; configure an appropriate
CUDA-enabled PyTorch installation separately when GPU training is required.
`neural.device` defaults to `cpu`; change it to `cuda` explicitly for GPU runs.

## Prepare real ApacheJIT data

1. Inspect the original dataset without extracting or running archive scripts:

   ```powershell
   .venv/Scripts/python.exe -m jit_sdp audit
   ```

   Expected valid counts: total 106,674; published train 44,834; large test 30,111;
   small test 7,526. These subsets are never concatenated. The report includes the
   archive SHA-256, schema checks, labels, per-project counts, invalid records and duplicates.
   Tiny numeric serialization differences between published subsets are logged with
   tolerance `rel_tol=1e-12, abs_tol=1e-12`; IDs, labels and other metadata must match exactly.

2. Edit `configs/default.json` before acquiring data. `projects: []` selects all projects.
   `repositories` maps **exact CSV project IDs** to existing local Git checkout or mirror paths.
   Historical Hadoop identifiers require explicit mapping rather than a guessed GitHub URL.
   For example, after independently confirming the appropriate histories:

   ```json
   {
     "repositories": {
       "apache/hadoop": "D:/apache-history/hadoop.git"
     }
   }
   ```

   Use project IDs from `data/archive_audit.json`; the example is not a claim that all
   Hadoop identifiers share a single history. The mapping is retained in the run configuration.
   To download non-Hadoop repositories absent from the mapping:

   ```powershell
   .venv/Scripts/python.exe -m jit_sdp acquire
   ```

   Acquisition creates full Git mirrors, which may require substantial disk space and time.
   It does not fetch into or modify an existing user-mapped checkout.

3. Retrieve complete messages, parents, committer timestamps and parent-relative patches:

   ```powershell
   .venv/Scripts/python.exe -m jit_sdp enrich
   ```

   `enrich --limit 100` processes a bounded number of uncached commits for an acquisition
   check. It is **not** a representative final cohort. `--retry-errors` retries cached failures.
   Retrieval uses local Git, with three bounded attempts on Git/timeout failures; errors and
   excluded roots, merges and unusable changes are saved rather than hidden.

4. Clean and save the eligible cohort. Run the scripts from the project root with the
   project environment active (`.venv/Scripts/Activate.ps1` in PowerShell). Each script
   reads `configs/default.json` automatically.

   ```powershell
   python clean_dataset.py
   ```

   The common cohort requires validated metrics and successfully retrieved one-parent
   textual code changes. The parser checks hunk lengths and retains file/hunk boundaries,
   display order, additions, deletions and context. Raw patches are saved separately.
   Empty messages remain valid; missing messages never receive fabricated content.
   Valid addition-only and deletion-only changes remain eligible.

   Cleaning saves `data/cleaned/dataset.jsonl`, `exclusions.json`, and
   `data_quality.json` under the configured `work_dir`. The reports record excluded
   rows and reasons, project and label counts, empty messages, and filtered files.
   Run enrichment first to populate the Git cache. Missing cache entries are reported
   as exclusions. Cleaning ends after writing the dataset and reports.

5. Freeze the training, validation, and test sets from the saved cleaned cohort.

   ```powershell
   python prepare_splits.py
   ```

   All published training records are ordered by **committer time**, with earlier 80% for
   training and later 20% for validation. Equal timestamps stay together. Published large
   test membership is preserved, subject only to documented eligibility exclusions.
   Training and validation must each contain both classes. Author dates are never substituted.

   If any published training commit reaches the earliest retained test timestamp, preparation
   stops. To adopt the predeclared exclusion rule, set
   `"temporal_overlap": "exclude_late_training"` **before final experiments**.
   Late training records are logged and excluded, never moved into test.

   Partition contents, IDs and configuration have checksums. Rebuilding a different cohort
   in the same `work_dir` is rejected; select a new work directory for a new cohort.
   Exact cross-partition message/code duplicates are reported in `content_overlap.json`.
   Near-duplicate investigation and historical label-availability audit remain research tasks.

## Train and compare

```powershell
python run_experiments.py
```

Edit the `models` and `seeds` lists in `configs/default.json` to select and order runs.
Both lists reject duplicates. The runner finishes all configured seeds for each model
before advancing to the next model. The default uses all three models and seeds 11, 23, 37.
CC2Vec uses commit-message terms to train a patch encoder, then fits a binary logistic head
on frozen patch vectors. Training and validation vectors are cached in CPU memory for head
fitting and threshold selection. Inference uses only the added and removed code channels.
The CLI's `compare` command regenerates reports from completed saved predictions.

The standalone comparison uses `experiment_v2` (`smoke_v2` for the smoke config).
Keep earlier run directories and their checkpoints and reports as historical records.

Use a different `run_id` for pilots. To use pilot settings with these scripts, apply
them in `configs/default.json`. `configs/smoke.json` reduces
model sizes and fitting budgets; it deliberately has a separate data directory and is not a
scientific final configuration. Run the synthetic integration test below to exercise it without
downloading Apache histories. Do not interpret fixture results as ApacheJIT performance.

All vocabularies, numeric scaling, representation learning and SMOTE fit on training only.
Neural checkpoints maximize validation MCC at 0.5, with earliest-epoch ties. Auxiliary
CC2Vec checkpoints minimize validation message-term loss. Early stopping has an independent
minimum-improvement rule. Final thresholds maximize validation MCC over 0.01…0.99,
breaking ties by closeness to 0.5 and then the smaller value. Test never selects settings.

Only one explicit neural configuration per method is fitted by this implementation.
This stays within the brief's three-configuration maximum; no unreported test-driven search
is performed. Adjust budgets after a pilot and freeze them before final testing.

Each run saves:

- Exact configuration, package and upstream source hashes, upstream Git revisions, library versions and hardware.
- Fitted dictionaries/scaler/vectorizer, classifier checkpoints, CC2Vec encoder, histories, validation threshold and runtime.
- Partition hashes, truncation counts, original and independent T1–T4 predictions, full encoded-input hashes and applicability statuses.
- Pooled and per-project MCC, AP, precision, recall, F1 and confusion counts; seed mean/sample SD and project macro means.
- Matched stress cohorts, score shifts, decision flips, MCC differences, project/commit bootstrap intervals and corrected paired tests.
- An atomic completion marker with artifact checksums; complete unchanged runs are skipped on resumption.

Reports are `runs/<run_id>/comparison.md` and `comparison.json`.
Single-class cohorts have undefined MCC/AP/ROC-AUC; constant predictions and zero predicted
positives are flagged. Three seeds are never treated as independent datasets for significance.
Bootstrap and permutation draws default to 1,000 and 9,999. Paired tests reuse each commit's
swap across seeds, with Holm correction. Project-block comparisons require at least six
nonzero project differences; a three-method Friedman test gates follow-up Wilcoxon tests.
Related projects, commit dependence and the small number of seeds limit inference.

Reload a trusted saved pipeline through `jit_sdp.runner.load_pipeline(run_folder)` or:

```powershell
.venv/Scripts/python.exe -m jit_sdp predict --run-folder runs/experiment_v2/cc2vec/seed_11 --input data/manifests/test.jsonl --output reports/reloaded.csv
```

This command expects canonical enriched records with labels/metrics, as used for evaluation.
Do not load joblib model files supplied by an untrusted source.

## Model adaptations and stress tests

Read `docs/methods.md` before interpreting results. Architectures are loaded directly from
the cloned DeepJIT and CC2Vec source; upstream files are not edited. The CC2Vec implementation
has documented correctness fixes and is **not a byte-for-byte reproduction** of its old runner.
JITLine is explicitly named an ApacheJIT adaptation (code tokens + available 12 metrics).
Training is from scratch on this split; external pretrained weights are not imported.

T1 reverses change polarity; T2 swaps old/new changed blocks; T3 inserts exactly
`<ADD> <DEL> <ADD> <DEL>` in the model-readable representation; T4 empties the message.
CC2Vec vectors are recomputed from each changed patch. T4 applies only to DeepJIT because
CC2Vec and JITLine have no message input at inference.
T1/T2 update JITLine's addition/deletion metrics only when patch and supplied metric scopes
reconcile. Encoded no-ops are separated from effective changes. Transformed labels remain
unknown, so MCC changes against original labels are diagnostics rather than verified accuracy.

## Verification

```powershell
.venv/Scripts/python.exe -m unittest discover -s tests -v
```

Checks cover the supplied archive, malformed diffs, real local Git retrieval, timestamp ties,
train-only vocabulary/unknown behavior, transformation integrity, metric edge cases, frozen
manifest tampering, encoder batch independence and a full synthetic three-model train/save/
reload/compare/resume cycle. Tests create disposable fixtures; they do not train on ApacheJIT.

## Layout

`src/jit_sdp/` contains the package; `configs/` holds experiment settings;
`tests/` holds protocol and integration tests. Generated `data/` contains untouched raw patches,
retrieval caches, archive audits and frozen manifests. Generated `runs/` holds model artifacts.
Training libraries are isolated in `.venv/`.

## Final experiment results

The [three-model comparison and verification audit](results/experiment_v5_v6/README.md) include the v5 and v6 results for seeds 11, 23 and 37.
