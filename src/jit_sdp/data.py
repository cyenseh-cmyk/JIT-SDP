"""Read original CSVs in-place inside the ZIP; never execute archive code."""
import csv
import hashlib
import io
import json
import math
import re
import sqlite3
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from .io import digest, file_hash, read_json, read_jsonl, write_json, write_jsonl

METRICS = "la ld nf nd ns ent ndev age nuc aexp arexp asexp".split()
FIELDS = ["commit_id", "project", "buggy", "fix", "year", "author_date", *METRICS]


def key(row):
    return row["project"] + ":" + row["commit_id"]


def boolean(value):
    if value not in ("True", "False"):
        raise ValueError(f"Invalid Boolean: {value!r}")
    return value == "True"


def validate(row):
    if not re.fullmatch(r"[0-9a-f]{40}", row["commit_id"]):
        raise ValueError("Invalid commit SHA")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", row["project"]):
        raise ValueError("Invalid project identifier")
    out = dict(row)
    out["buggy"], out["fix"] = boolean(row["buggy"]), boolean(row["fix"])
    for name in ["year", "author_date"]:
        out[name] = int(row[name])
    if not 1980 <= out["year"] <= 2100 or not 0 < out["author_date"] < 4102444800:
        raise ValueError("Implausible year or author timestamp")
    for name in METRICS:
        out[name] = float(row[name])
        if not math.isfinite(out[name]) or out[name] < 0:
            raise ValueError(f"Invalid metric {name}")
    return out


def archive_tables(archive):
    tables, audit = {}, {}
    with zipfile.ZipFile(archive) as source:
        for name in ("total", "train", "test_large", "test_small"):
            member = f"apachejit/dataset/apachejit_{name}.csv"
            with source.open(member) as stream:
                reader = csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig"))
                if reader.fieldnames != FIELDS:
                    raise ValueError(f"Unexpected schema: {member}")
                rows, duplicates, invalid = {}, [], []
                for index, row in enumerate(reader, start=2):
                    try:
                        clean = validate(row)
                    except (ValueError, TypeError) as exc:
                        invalid.append({"line": index, "reason": str(exc),
                                        "key": f"{row.get('project') or ''}:{row.get('commit_id') or ''}"})
                        continue
                    ident = key(clean)
                    if ident in rows:
                        if rows[ident] != clean:
                            raise ValueError(f"Conflicting duplicate: {ident}")
                        duplicates.append({"line": index, "key": ident})
                    rows[ident] = clean
            tables[name] = rows
            audit[name] = {"valid": len(rows), "buggy": sum(r["buggy"] for r in rows.values()),
                           "duplicates_removed": duplicates, "invalid_excluded": invalid,
                           "projects": dict(Counter(r["project"] for r in rows.values()))}
    for name in ("train", "test_large", "test_small"):
        precision_differences = []
        for ident, row in tables[name].items():
            total_row = tables["total"].get(ident)
            if total_row is None:
                raise ValueError(f"{name} does not match total: {ident}")
            differences = [field for field in FIELDS if total_row[field] != row[field]]
            if any(field not in METRICS or not math.isclose(total_row[field], row[field],
                    rel_tol=1e-12, abs_tol=1e-12) for field in differences):
                raise ValueError(f"{name} conflicts with total: {ident}")
            if differences:
                precision_differences.append({"key": ident, "fields": differences})
        audit[name]["numeric_precision_differences_from_total"] = precision_differences
    if set(tables["train"]) & set(tables["test_large"]):
        raise ValueError("Published training/test overlap")
    if not set(tables["test_small"]) <= set(tables["test_large"]):
        raise ValueError("Small test is not a subset of large test")
    return tables, audit


def audit_archive(cfg):
    tables, report = archive_tables(cfg["archive"])
    report["archive"] = {"path": cfg["archive"], "sha256": file_hash(cfg["archive"]),
                         "source": "https://zenodo.org/records/5907847", "version": 2}
    write_json(Path(cfg["work_dir"]) / "archive_audit.json", report)
    return tables


def split_records(records, validation_fraction=0.2, overlap="error"):
    train = sorted((r for r in records if r["published_partition"] == "train"),
                   key=lambda r: (r["committer_time"], r["project"], r["commit_id"]))
    test = [r for r in records if r["published_partition"] == "test"]
    if len(train) < 2 or not test:
        raise ValueError("Need at least two training commits and a nonempty published test set")
    cutoff = min(r["committer_time"] for r in test)
    late = [r for r in train if r["committer_time"] >= cutoff]
    if late and overlap == "error":
        raise ValueError(f"Temporal overlap: {len(late)} training commits reach test time. "
                         "Choose exclude_late_training explicitly before freezing the experiment.")
    if late:
        train = [r for r in train if r["committer_time"] < cutoff]
    if len(train) < 2:
        raise ValueError("Too few training records after chronological exclusions")
    index = min(len(train) - 1, max(1, int(len(train) * (1 - validation_fraction))))
    boundary = train[index]["committer_time"]
    groups = {"train": [r for r in train if r["committer_time"] < boundary],
              "validation": [r for r in train if r["committer_time"] >= boundary], "test": test}
    for name, group in groups.items():
        if not group:
            raise ValueError(f"Empty {name} partition; equal timestamps cannot be split")
        if name != "test" and len({r["buggy"] for r in group}) != 2:
            raise ValueError(f"{name} needs both classes for checkpoint and threshold selection")
    return groups, [key(r) for r in late]


def clean_dataset(cfg):
    tables = audit_archive(cfg)
    cache = Path(cfg["work_dir"]) / "cache"
    database = cache.parent / "enrichment.sqlite3"
    if database.exists():
        from contextlib import closing
        with closing(sqlite3.connect(database)) as db:
            return clean_from_cache(cfg, tables, cache, db)
    return clean_from_cache(cfg, tables, cache)


def eligible_records(cfg, tables, cache, exclusions, db):
    for partition, table in (("train", tables["train"]), ("test", tables["test_large"])):
        for ident, original in sorted(table.items()):
            if cfg["projects"] and original["project"] not in cfg["projects"]:
                exclusions.append({"key": ident, "reason": "project_selection"})
                continue
            path = cache / original["project"] / (original["commit_id"] + ".json")
            stored = None if db is None else db.execute(
                "SELECT record FROM enrichment WHERE project = ? AND sha = ?",
                (original["project"], original["commit_id"])).fetchone()
            if stored is not None:
                enriched = json.loads(stored[0])
            elif path.exists():
                enriched = read_json(path)
            else:
                exclusions.append({"key": ident, "reason": "not_retrieved"})
                continue
            if enriched.get("retrieval_status") != "ok" or enriched.get("parse_status") != "ok":
                exclusions.append({"key": ident, "reason": enriched.get("reason", "invalid_cache")})
                continue
            if enriched.get("commit_id") != original["commit_id"] or enriched.get("project") != original["project"]:
                raise ValueError(f"Cache identity mismatch: {ident}")
            if len(enriched.get("parent_ids", [])) != 1 or not enriched.get("files"):
                raise ValueError(f"Invalid eligible cache: {ident}")
            if not isinstance(enriched.get("message"), str) or type(enriched.get("committer_time")) is not int:
                raise ValueError(f"Missing message/committer timestamp: {ident}")
            if not 0 < enriched["committer_time"] < 4102444800:
                raise ValueError(f"Implausible committer timestamp: {ident}")
            # Cache payloads cannot replace source labels, metrics or metadata.
            for field in FIELDS:
                if field in enriched and enriched[field] != original[field]:
                    raise ValueError(f"Cached {field} disagrees with source: {ident}")
            yield {**enriched, **original, "published_partition": partition}


def clean_from_cache(cfg, tables, cache, db=None):
    exclusions = []
    work_dir = Path(cfg["work_dir"])
    audit = read_json(work_dir / "archive_audit.json")
    archive_exclusions = [
        {"table": table, "kind": kind, **entry}
        for table in tables
        for kind in ("duplicates_removed", "invalid_excluded")
        for entry in audit[table][kind]
    ]
    quality = {
        "archive_sha256": audit["archive"]["sha256"],
        "selected_projects": cfg["projects"],
        "published_candidates": len(tables["train"]) + len(tables["test_large"]),
        "clean_records": 0,
        "archive_exclusions": len(archive_exclusions),
        "records_by_label": {"buggy": 0, "clean": 0},
        "empty_messages": 0,
        "filtered_files": 0,
    }
    projects = Counter()

    def counted_records():
        for row in eligible_records(cfg, tables, cache, exclusions, db):
            projects[row["project"]] += 1
            quality["clean_records"] += 1
            quality["records_by_label"]["buggy" if row["buggy"] else "clean"] += 1
            quality["empty_messages"] += not row["message"]
            quality["filtered_files"] += row.get("filtered_file_count", 0)
            yield row

    out = work_dir / "cleaned"
    write_jsonl(out / "dataset.jsonl", counted_records())
    quality["records_by_project"] = dict(projects)
    quality["excluded_candidates"] = len(exclusions)
    quality["exclusions_by_reason"] = dict(Counter(item["reason"] for item in exclusions))
    write_json(out / "exclusions.json", {"archive": archive_exclusions, "cohort": exclusions})
    quality["dataset_sha256"] = file_hash(out / "dataset.jsonl")
    quality["exclusions_sha256"] = file_hash(out / "exclusions.json")
    write_json(out / "data_quality.json", quality)
    return quality


def prepare_splits(cfg):
    cleaned = Path(cfg["work_dir"]) / "cleaned"
    quality = read_json(cleaned / "data_quality.json")
    if quality["selected_projects"] != cfg["projects"] or quality["archive_sha256"] != file_hash(cfg["archive"]):
        raise ValueError("Archive or project selection changed. Run clean_dataset.py again.")
    for filename, checksum in (("dataset.jsonl", "dataset_sha256"), ("exclusions.json", "exclusions_sha256")):
        if file_hash(cleaned / filename) != quality[checksum]:
            raise ValueError(f"Modified cleaning output: {filename}")
    records = []
    dataset = cleaned / "dataset.jsonl"
    with dataset.open("rb") as stream:
        offset = 0
        for line in stream:
            if line.strip():
                row = json.loads(line)
                metadata = {field: row[field] for field in (
                    "project", "commit_id", "published_partition", "committer_time", "buggy")}
                records.append({**metadata, "offset": offset})
            offset += len(line)
    if not records:
        raise ValueError("No eligible commits. Enrich the Git cache and rerun clean_dataset.py.")
    exclusions = read_json(cleaned / "exclusions.json")["cohort"]
    groups, late = split_records(records, cfg["validation_fraction"], cfg["temporal_overlap"])
    exclusions += [{"key": ident, "reason": "temporal_overlap"} for ident in late]
    manifest = {"archive_sha256": quality["archive_sha256"], "policy": {
        "validation_fraction": cfg["validation_fraction"], "temporal_overlap": cfg["temporal_overlap"],
        "projects": cfg["projects"], "parser_version": 1}, "partitions": {}, "exclusions": exclusions}
    out = Path(cfg["work_dir"]) / "manifests"
    out.parent.mkdir(parents=True, exist_ok=True)
    seen, duplicate_content = {}, []
    with tempfile.TemporaryDirectory(dir=out.parent) as staging, dataset.open("rb") as stream:
        staging = Path(staging)
        for name, group in groups.items():
            group.sort(key=key)
            checksum = hashlib.sha256(b"[")
            with (staging / (name + ".jsonl")).open("wb") as partition:
                for index, metadata in enumerate(group):
                    stream.seek(metadata["offset"])
                    record = json.loads(stream.readline())
                    encoded = json.dumps(record, sort_keys=True, allow_nan=False).encode()
                    if index:
                        checksum.update(b", ")
                    checksum.update(encoded)
                    partition.write(encoded + b"\n")
                    content = digest([record["message"], record["files"]])
                    ident = key(record)
                    if content in seen and seen[content][0] != name:
                        duplicate_content.append({"first": seen[content], "other": [name, ident]})
                    seen[content] = [name, ident]
            checksum.update(b"]")
            manifest["partitions"][name] = {
                "keys": [key(r) for r in group], "content_hash": checksum.hexdigest(),
                "count": len(group), "buggy": sum(r["buggy"] for r in group),
                "projects": dict(Counter(r["project"] for r in group)),
                "time_min": min(r["committer_time"] for r in group),
                "time_max": max(r["committer_time"] for r in group)}
        manifest["hash"] = digest(manifest)
        if (out / "split.json").exists() and read_json(out / "split.json") != manifest:
            raise ValueError("Frozen manifest differs; use a new work_dir for a new cohort")
        write_json(staging / "content_overlap.json", duplicate_content)
        write_json(staging / "split.json", manifest)
        out.mkdir(exist_ok=True)
        for filename in (*(name + ".jsonl" for name in groups), "content_overlap.json", "split.json"):
            (staging / filename).replace(out / filename)
    return manifest


def prepare(cfg):
    """Preserve the combined cleaning and splitting API used by the CLI."""
    clean_dataset(cfg)
    return prepare_splits(cfg)


def load_partitions(cfg):
    root = Path(cfg["work_dir"]) / "manifests"
    manifest = read_json(root / "split.json")
    claimed = manifest["hash"]
    if digest({k: v for k, v in manifest.items() if k != "hash"}) != claimed:
        raise ValueError("Manifest checksum mismatch")
    groups = {name: list(read_jsonl(root / (name + ".jsonl"))) for name in manifest["partitions"]}
    all_keys = []
    for name, group in groups.items():
        if digest(group) != manifest["partitions"][name]["content_hash"]:
            raise ValueError(f"Modified partition: {name}")
        all_keys.extend(map(key, group))
    if len(all_keys) != len(set(all_keys)):
        raise ValueError("Partitions overlap")
    return groups, manifest
