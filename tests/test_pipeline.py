"""Meaningful protocol checks and a complete three-method synthetic integration test."""
import copy
import csv
import io
import json
import runpy
import shutil
from unittest.mock import patch
import subprocess
import tempfile
import unittest
import zipfile
from pathlib import Path

from jit_sdp.config import load
from jit_sdp.data import FIELDS, METRICS, archive_tables, key, load_partitions, prepare, split_records, validate
from jit_sdp.features import NeuralAdapter, JITLineAdapter
from jit_sdp.gitdata import parse_patch, retrieve, enrich
from jit_sdp.io import read_json, write_json
from jit_sdp.metrics import classification, select_threshold
from jit_sdp.stress import transform

ROOT = Path(__file__).resolve().parents[1]


def record(index, partition="train"):
    added = "if (count >= limit) return result;" if index % 2 else "return safe;"
    patch = f"diff --git a/A.java b/A.java\n--- a/A.java\n+++ b/A.java\n@@ -1 +1 @@\n-return old;\n+{added}\n"
    return {"project": "apache/fixture", "commit_id": f"{index + 1:040x}",
        "buggy": bool(index % 2), "fix": False, "year": 2010, "author_date": 1262304000 + index,
        **{m: 1.0 for m in METRICS}, "message": "repair boundary" if index % 2 else "clean refactor",
        "committer_time": 1262304000 + index, "parent_ids": ["a" * 40],
        "retrieval_status": "ok", "parse_status": "ok", "parser_version": 1,
        "files": parse_patch(patch), "filtered_file_count": 0, "published_partition": partition}


def fixture_archive(path, records):
    groups = {"total": records, "train": [r for r in records if r["published_partition"] == "train"],
              "test_large": [r for r in records if r["published_partition"] == "test"]}
    groups["test_small"] = groups["test_large"][:2]
    with zipfile.ZipFile(path, "w") as archive:
        for name, rows in groups.items():
            text = io.StringIO(newline="")
            writer = csv.DictWriter(text, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({field: row[field] for field in FIELDS} for row in rows)
            archive.writestr(f"apachejit/dataset/apachejit_{name}.csv", text.getvalue())


class ProtocolTests(unittest.TestCase):
    def test_friedman_uses_models_as_samples(self):
        import numpy as np
        from scipy.stats import friedmanchisquare
        from jit_sdp.analysis import friedman_model_pvalue

        # Constant scores within each model still differ within every project.
        matrix = np.array([[1] * 6, [2] * 6, [3] * 6])
        self.assertAlmostEqual(friedman_model_pvalue(matrix), friedmanchisquare(*matrix).pvalue)

    def test_friedman_skips_identical_models_with_varying_projects(self):
        import numpy as np
        from jit_sdp.analysis import friedman_model_pvalue

        with patch("scipy.stats.friedmanchisquare") as test:
            self.assertIsNone(friedman_model_pvalue(np.tile(np.arange(6), (3, 1))))
            test.assert_not_called()

    def test_config_rejects_duplicate_models_and_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            for name, entries in (("models", ["deepjit", "deepjit"]), ("seeds", [11, 11])):
                with self.subTest(name=name):
                    write_json(path, {name: entries})
                    with self.assertRaisesRegex(ValueError, "unique"):
                        load(path)

    def test_patch_contents_and_reject_truncation(self):
        parsed = record(1)["files"]
        self.assertEqual(parsed[0]["new_path"], "A.java")
        self.assertEqual([x["kind"] for x in parsed[0]["hunks"][0]["lines"]], ["-", "+"])
        with self.assertRaises(ValueError):
            parse_patch("diff --git a/A b/A\n--- a/A\n+++ b/A\n@@ -1 +1 @@\n-old\n")
        # A changed line beginning with header-like punctuation remains code.
        patch = "diff --git a/A b/A\n--- a/A\n+++ b/A\n@@ -1 +1 @@\n----x\n++++x\n"
        self.assertEqual(parse_patch(patch)[0]["hunks"][0]["lines"][0]["text"], "---x")

    def test_labels_and_fractional_metrics(self):
        row = {f: str(record(1)[f]) for f in FIELDS}
        row["ndev"] = "0.25"
        self.assertEqual(validate(row)["ndev"], 0.25)
        row["buggy"] = "perhaps"
        with self.assertRaises(ValueError):
            validate(row)

    def test_training_only_dictionary_and_empty_message(self):
        cfg = load(ROOT / "configs/smoke.json")
        adapter = NeuralAdapter(cfg["encoding"]).fit([record(0), record(1)])
        unseen = record(2)
        unseen["message"] = "NeverSeenToken"
        encoded = adapter.encode(unseen)
        self.assertEqual(int(encoded["message"][0]), 1)
        self.assertNotIn("neverseentoken", adapter.state["message"])
        unseen["message"] = ""
        self.assertFalse(adapter.encode(unseen)["message"].any())

    def test_temporal_split_and_ties(self):
        rows = [record(i) for i in range(10)] + [record(20 + i, "test") for i in range(4)]
        rows[7]["committer_time"] = rows[8]["committer_time"]
        groups, _ = split_records(rows)
        self.assertLess(max(r["committer_time"] for r in groups["train"]),
                        min(r["committer_time"] for r in groups["validation"]))
        rows[-1]["committer_time"] = rows[0]["committer_time"]
        with self.assertRaisesRegex(ValueError, "Temporal overlap"):
            split_records(rows)

    def test_stress_no_mutation_involution_scope(self):
        original = record(1)
        snapshot = copy.deepcopy(original)
        for condition in ("T1", "T2"):
            changed, reason, _ = transform(original, condition, "deepjit")
            restored, _, _ = transform(changed, condition, "deepjit")
            self.assertFalse(reason)
            self.assertEqual(restored, original)
        self.assertEqual(original, snapshot)
        _, reason, _ = transform(original, "T4", "jitline_apachejit")
        self.assertEqual(reason, "no_message_channel")
        original["la"] = 9
        _, reason, _ = transform(original, "T1", "jitline_apachejit")
        self.assertEqual(reason, "metric_scope_mismatch")
        one_sided = record(2)
        one_sided["files"][0]["hunks"][0]["lines"] = [{"kind": "+", "text": "new"}]
        self.assertEqual(transform(one_sided, "T2", "deepjit")[1], "no_two_sided_hunk")

    def test_t2_multiple_regions_and_unequal_sides(self):
        original = record(1)
        original["files"][0]["hunks"][0]["lines"] = [
            {"kind": "-", "text": "old1"}, {"kind": "-", "text": "old2"},
            {"kind": "+", "text": "new1"}, {"kind": " ", "text": "context"},
            {"kind": "-", "text": "old3"}, {"kind": "+", "text": "new2"},
            {"kind": "+", "text": "new3"}, {"kind": " ", "text": "context2"},
            {"kind": "+", "text": "new4"}]
        original["la"], original["ld"] = 4, 3
        changed, reason, count = transform(original, "T2", "jitline_apachejit")
        restored, _, _ = transform(changed, "T2", "jitline_apachejit")
        self.assertEqual(restored, original)
        self.assertEqual((changed["la"], changed["ld"]), (3, 4))
        self.assertEqual(count, 1)
        self.assertFalse(reason)

    def test_noop_is_adapter_specific(self):
        cfg = load(ROOT / "configs/smoke.json")
        original = record(1)
        adapter = NeuralAdapter(cfg["encoding"]).fit([record(0), original])
        changed, _, _ = transform(original, "T1", "deepjit")
        self.assertEqual(adapter.input_hash(original, "deepjit"), adapter.input_hash(changed, "deepjit"))
        self.assertNotEqual(adapter.input_hash(original, "cc2vec"), adapter.input_hash(changed, "cc2vec"))
        forest = JITLineAdapter(min_df=1).fit([record(0), original])
        self.assertEqual(forest.input_hash(original), forest.input_hash(changed))

    def test_cc2vec_inference_encoding_uses_only_patch_channels(self):
        cfg = load(ROOT / "configs/smoke.json")
        original = record(0)
        adapter = NeuralAdapter(cfg["encoding"]).fit([original, record(1)])
        changed = {**original, "message": "repair " * 100}
        names = adapter.input_names("cc2vec")
        self.assertEqual(set(adapter.encode(changed, names)), {"added", "removed", "truncated"})
        self.assertFalse(adapter.encode(changed, names)["truncated"])
        self.assertTrue(adapter.encode(changed, adapter.input_names("deepjit"))["truncated"])
        self.assertEqual(adapter.input_hash(original, "cc2vec"), adapter.input_hash(changed, "cc2vec"))
        self.assertEqual(transform(original, "T4", "cc2vec")[1], "no_message_channel")

    def test_cc2vec_t3_prefix_reaches_first_nonblank_file(self):
        cfg = load(ROOT / "configs/smoke.json")
        original = record(1)
        original["files"].insert(0, {"hunks": [{"lines": [{"kind": "+", "text": " "}]}]})
        adapter = NeuralAdapter(cfg["encoding"]).fit([record(0), original])
        changed, reason, _ = transform(original, "T3", "cc2vec")
        self.assertEqual(reason, "")
        names = adapter.input_names("cc2vec")
        before = adapter.encode(original, names)
        after = adapter.encode(changed, names)
        prefix_ids = [adapter.state["code"].get(token, 1)
                      for token in ("<", "add", ">", "<", "del", ">", "<", "add")]
        self.assertEqual(after["removed"][1, 0].tolist(), prefix_ids)
        self.assertNotEqual(before["removed"][1, 0].tolist(), after["removed"][1, 0].tolist())
        self.assertEqual(before["added"].tolist(), after["added"].tolist())

    def test_metrics_degenerate_and_threshold(self):
        perfect = classification([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(perfect["mcc"], 1.0)
        self.assertEqual((perfect["tn"], perfect["tp"]), (2, 2))
        self.assertIsNone(classification([1, 1], [0.1, 0.2])["ap"])
        self.assertEqual(classification([0, 1], [0.0, 0.0])["mcc"], 0.0)
        self.assertEqual(select_threshold([0, 1], [0.1, 0.9]), 0.5)

    def test_smote_training_only_and_fit_cap(self):
        from jit_sdp.forest import tune, ForestPipeline
        cfg = load(ROOT / "configs/smoke.json")
        cfg["jitline"]["population"] = 4
        cfg["jitline"]["candidate_evaluations"] = 5
        train = [record(i * 2) for i in range(8)] + [record(i * 2 + 1) for i in range(3)]
        validation = [record(40), record(41), record(42), record(43)]
        adapter = JITLineAdapter(min_df=1).fit(train)
        original_validation = adapter.transform(validation).copy()
        selected, candidates = tune(cfg, adapter, train, validation, 11)
        self.assertEqual(len(candidates), 5)
        self.assertLessEqual(selected["neighbors"], 2)
        self.assertEqual((adapter.transform(validation) != original_validation).nnz, 0)

    def test_empty_cohort_and_multiple_seed_pairs(self):
        from jit_sdp.analysis import paired_arrays, uncertainty, holm
        import numpy as np
        self.assertIsNone(paired_arrays([]))
        rows = []
        for seed in (11, 23):
            for i in range(4):
                rows.append({"seed": seed, "project": "fixture", "commit_id": str(i), "y_original": i % 2,
                             "yhat_original": i % 2, "yhat_transformed": 1 - i % 2})
        result = uncertainty(rows, {"bootstrap_draws": 20, "permutation_draws": 19}, np.random.default_rng(1))
        self.assertEqual(result["delta_mcc"], 2)
        self.assertEqual(result["n"], 4)
        entries = [{"p": 0.01}, {"p": 0.04}, {"p": None}]
        holm(entries)
        self.assertEqual(entries[0]["p_holm"], 0.02)
        self.assertEqual(entries[1]["p_holm"], 0.04)

    def test_all_neural_stages_can_overfit_tiny_training_batch(self):
        import torch
        from jit_sdp.neural import batch_input, classifier, representation, seed_everything
        cfg = load(ROOT / "configs/smoke.json")
        cfg["neural"]["dropout"] = 0.0
        seed_everything(11, 1)
        rows = [record(0), record(1)]
        adapter = NeuralAdapter(cfg["encoding"]).fit(rows)
        batch = batch_input(adapter, rows, "cpu")
        encoder = representation(cfg, adapter.state)
        target = torch.zeros(2, len(adapter.state["message"]))
        target.scatter_(1, batch["message"], 1.0)

        def optimize(model, closure):
            optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
            initial = float(closure().detach())
            for _ in range(25):
                optimizer.zero_grad()
                loss = closure()
                loss.backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()
            self.assertLess(float(closure().detach()), initial * 0.8)

        optimize(encoder, lambda: torch.nn.functional.binary_cross_entropy_with_logits(
            encoder.logits(batch["added"], batch["removed"])[:, 2:], target[:, 2:]))
        encoder.requires_grad_(False).eval()
        feature = encoder.features(batch["added"], batch["removed"]).detach()
        labels = torch.tensor([0.0, 1.0])
        deepjit = classifier(cfg, adapter.state, "deepjit")
        optimize(deepjit, lambda: torch.nn.functional.binary_cross_entropy(
            deepjit(batch["message"], batch["code"]), labels))
        cc2vec = classifier(cfg, adapter.state, "cc2vec")
        optimize(cc2vec, lambda: torch.nn.functional.binary_cross_entropy(cc2vec(feature), labels))

    def test_full_archive_counts(self):
        tables, _ = archive_tables(ROOT / "apachejit_dataset_replication.zip")
        self.assertEqual({k: len(v) for k, v in tables.items()},
                         {"total": 106674, "train": 44834, "test_large": 30111, "test_small": 7526})

    def test_git_retrieval_real_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            def command(*args):
                return subprocess.check_output(["git", "-C", tmp, *args], text=True).strip()
            command("init", "-q")
            command("config", "user.name", "Fixture")
            command("config", "user.email", "fixture@example.invalid")
            (repo / "A.java").write_text("class Old {}\n")
            command("add", "A.java")
            command("commit", "-qm", "first")
            (repo / "A.java").write_text("class New {}\n")
            command("add", "A.java")
            command("commit", "-qm", "second")
            sha = command("rev-parse", "HEAD")
            result = retrieve(repo, "apache/fixture", sha, repo / "raw.patch")
            self.assertEqual(result["message"], "second")
            self.assertEqual(result["files"][0]["hunks"][0]["lines"][0]["text"], "class Old {}")


class IntegrationTests(unittest.TestCase):
    def test_cc2vec_checkpoint_reloads_for_a_single_patch(self):
        import numpy as np
        from jit_sdp.neural import NeuralPipeline
        from jit_sdp.runner import prediction_rows

        cfg = load(ROOT / "configs/smoke.json")
        cfg["cc2vec"]["max_epochs"] = cfg["neural"]["max_epochs"] = 1
        train = [record(i) for i in range(4)]
        validation = [record(i) for i in (4, 5)]
        adapter = NeuralAdapter(cfg["encoding"]).fit(train)
        patch_only = record(7, "test")
        del patch_only["message"]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            fitted = NeuralPipeline(cfg, adapter, "cc2vec", 11).fit(train, validation, out)
            restored = NeuralPipeline(cfg, adapter, "cc2vec", 11).load(out)
            self.assertEqual({path.name for path in out.iterdir()}, {"encoder.pt", "classifier.pt"})
            rows = prediction_rows(cfg, restored, [patch_only], "cc2vec", 11, 0.5)
            self.assertEqual([row["condition"] for row in rows], ["original", "T1", "T2", "T3", "T4"])
            self.assertTrue(all(0 <= row["p_transformed"] <= 1 for row in rows[:-1]))
            self.assertIsNone(rows[-1]["p_transformed"])
            np.testing.assert_allclose(restored.predict([patch_only]), fitted.predict([patch_only]), atol=1e-7)

    def test_three_models_train_save_reload_compare(self):
        from jit_sdp.runner import run, load_pipeline, prediction_rows
        from jit_sdp.neural import batch_input, representation, NeuralPipeline
        import numpy as np
        import torch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            overrides = json.loads((ROOT / "configs/smoke.json").read_text())
            overrides.update(archive=str(root / "dataset.zip"), work_dir=str(root / "data"),
                             run_dir=str(root / "runs"), baseline_dir=str(ROOT / "baseline implementations"))
            config_path = root / "configs/default.json"
            write_json(config_path, overrides)
            cfg = load(config_path)
            records = [record(i) for i in range(20)] + [record(i, "test") for i in range(30, 36)]
            fixture_archive(cfg["archive"], records)
            for row in records:
                write_json(Path(cfg["work_dir"]) / "cache" / row["project"] / (row["commit_id"] + ".json"), row)
            for name in ("clean_dataset.py", "prepare_splits.py", "run_experiments.py"):
                shutil.copyfile(ROOT / name, root / name)
            runpy.run_path(str(root / "clean_dataset.py"), run_name="__main__")
            cleaned = Path(cfg["work_dir"]) / "cleaned"
            self.assertEqual(read_json(cleaned / "data_quality.json")["clean_records"], len(records))
            self.assertEqual(read_json(cleaned / "exclusions.json"), {"archive": [], "cohort": []})
            self.assertFalse((Path(cfg["work_dir"]) / "manifests").exists())
            self.assertFalse(Path(cfg["run_dir"]).exists())
            runpy.run_path(str(root / "prepare_splits.py"), run_name="__main__")
            groups, _ = load_partitions(cfg)
            adapter = NeuralAdapter(cfg["encoding"]).fit(groups["train"])
            # Fixed encoder extraction must be independent of batch composition.
            encoder = representation(cfg, adapter.state).eval()
            both = batch_input(adapter, records[:2], "cpu")
            one = batch_input(adapter, records[:1], "cpu")
            with torch.no_grad():
                np.testing.assert_allclose(encoder.features(both["added"], both["removed"])[0].numpy(),
                    encoder.features(one["added"], one["removed"])[0].numpy(), atol=1e-6)
            feature_passes = []
            original_cache = NeuralPipeline._cache_features

            def cache_features(pipeline, rows):
                feature_passes.append(len(rows))
                return original_cache(pipeline, rows)

            with patch.object(NeuralPipeline, "_cache_features", cache_features):
                runpy.run_path(str(root / "run_experiments.py"), run_name="__main__")
            self.assertEqual(feature_passes, [len(groups["train"]), len(groups["validation"])])
            report = read_json(Path(cfg["run_dir"]) / cfg["run_id"] / "comparison.json")
            self.assertEqual(len(report["seed_summary"]), 15)
            self.assertEqual(cfg["models"], ["deepjit", "cc2vec", "jitline_apachejit"])
            self.assertEqual({row["model"] for row in report["seed_summary"]}, set(cfg["models"]))
            for model in cfg["models"]:
                out = Path(cfg["run_dir"]) / cfg["run_id"] / model / "seed_11"
                pipeline, threshold = load_pipeline(out)
                first = pipeline.predict(groups["test"])
                second = pipeline.predict(groups["test"])
                np.testing.assert_allclose(first, second, atol=1e-7)
                with (out / "predictions.csv").open(newline="", encoding="utf-8") as stream:
                    predictions = list(csv.DictReader(stream))
                self.assertEqual(len(predictions), 5 * len(groups["test"]))
                self.assertEqual({row["condition"] for row in predictions}, {"original", "T1", "T2", "T3", "T4"})
                saved = {row["commit_id"]: float(row["p_original"]) for row in predictions
                         if row["condition"] == "original"}
                np.testing.assert_allclose(first, [saved[row["commit_id"]] for row in groups["test"]], atol=1e-7)
                t4 = [row for row in predictions if row["condition"] == "T4"]
                if model in ("cc2vec", "jitline_apachejit"):
                    self.assertTrue(all(row["status"] == "structural_na" and
                                        row["status_reason"] == "no_message_channel" and
                                        not row["p_transformed"] and not row["input_hash_transformed"] for row in t4))
                else:
                    self.assertTrue(all(row["status"] == "effective_change" and row["p_transformed"] for row in t4))
                artifacts = read_json(out / "complete.json")["artifacts"]
                if model == "cc2vec":
                    self.assertTrue({"adapter.json", "encoder.pt", "classifier.pt"} <= artifacts.keys())
                    head = torch.load(out / "classifier.pt", weights_only=True)
                    self.assertEqual({tuple(tensor.shape) for tensor in head.values()},
                                     {(1, 3 * cfg["cc2vec"]["embed_size"] + 4), (1,)})
                    edits = [row for row in predictions if row["condition"] in ("T1", "T2", "T3")]
                    self.assertTrue(all(row["status"] == "effective_change" for row in edits))
                    without_messages = [{k: v for k, v in row.items() if k != "message"} for row in groups["test"]]
                    np.testing.assert_allclose(pipeline.predict(without_messages), first, atol=1e-7)
                elif model == "deepjit":
                    self.assertTrue({"adapter.json", "classifier.pt"} <= artifacts.keys())
                    self.assertNotIn("encoder.pt", artifacts)
                else:
                    self.assertIn("forest.joblib", artifacts)
                regenerated = prediction_rows(cfg, pipeline, groups["test"], model, 11, threshold)
                for saved_row, fresh_row in zip(predictions, regenerated, strict=True):
                    for field in ("model", "commit_id", "condition", "status", "status_reason",
                                  "input_hash_original", "input_hash_transformed", "yhat_transformed"):
                        value = fresh_row[field]
                        self.assertEqual(saved_row[field], "" if value is None else str(value))
                    if fresh_row["p_transformed"] is not None:
                        self.assertAlmostEqual(float(saved_row["p_transformed"]), fresh_row["p_transformed"], places=7)
                self.assertGreater(threshold, 0)
            run(cfg)  # completed runs are checksum-verified and skipped
            path = Path(cfg["work_dir"]) / "manifests/train.jsonl"
            path.write_text(path.read_text() + "{}\n")
            with self.assertRaisesRegex(ValueError, "Modified partition"):
                load_partitions(cfg)


if __name__ == "__main__":
    unittest.main()
