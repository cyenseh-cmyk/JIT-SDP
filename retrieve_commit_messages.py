"""Export ApacheJIT commit messages by project and exact commit ID."""
import csv
import io
import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jit_sdp.config import load
from jit_sdp.data import archive_tables
from jit_sdp.io import write_json


HADOOP_HISTORY = {"apache/hadoop-hdfs", "apache/hadoop-mapreduce"}


def read_messages(history, ids):
    # Query exact objects, including commits that are no longer on a named branch.
    result = subprocess.run(
        ["git", "--no-lazy-fetch", "-C", str(history), "cat-file", "--batch"],
        input=("\n".join(sorted(ids)) + "\n").encode("ascii"),
        capture_output=True, check=True, timeout=600)
    stream = io.BytesIO(result.stdout)
    messages = {}
    while header := stream.readline():
        fields = header.split()
        if len(fields) == 2 and fields[1] == b"missing":
            continue
        sha, kind, size = fields
        obj = stream.read(int(size))
        if stream.read(1) != b"\n":
            raise ValueError("Invalid Git batch output")
        if kind != b"commit":
            continue
        metadata, separator, body = obj.partition(b"\n\n")
        if not separator:
            raise ValueError(f"Invalid Git commit object: {sha.decode('ascii')}")
        encoding = next((line[9:].decode("ascii") for line in metadata.splitlines()
                         if line.startswith(b"encoding ")), "utf-8")
        try:
            message = body.decode(encoding, errors="replace")
        except LookupError:
            message = body.decode("utf-8", errors="replace")
        messages[sha.decode("ascii")] = message.rstrip("\n")
    return messages


def retrieve_project(project, ids, cfg):
    name = project.replace("/", "__")
    root = Path(cfg["work_dir"])
    existing = root / "message_histories" / name
    history = root / "message_sources" / name
    configured = cfg["repositories"].get(project)
    source = f"https://github.com/{project}.git"
    try:
        if configured:
            history = Path(configured)
            source = str(history)
        elif existing.exists() and any((existing / "objects/pack").glob("*.idx")):
            history = existing
        elif not history.exists():
            history.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--bare", "--filter=tree:0", source, str(history)],
                           check=True, timeout=3600)
        messages = read_messages(history, ids)
        error = None
    except (OSError, subprocess.SubprocessError) as exc:
        messages, error = {}, str(exc)
    print(f"{project}: {len(messages)}/{len(ids)} messages", flush=True)
    return project, messages, source, error


def main():
    cfg = load(Path(__file__).resolve().parent / "configs/default.json")
    tables, _ = archive_tables(cfg["archive"])
    ids = defaultdict(set)
    for record in tables["total"].values():
        ids[record["project"]].add(record["commit_id"])
    ids["apache/hadoop"].update(*(ids[project] for project in HADOOP_HISTORY))
    output = Path(cfg["work_dir"]) / "commit_messages_verified"
    output.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict((project, (messages, source, error)) for project, messages, source, error in
                       pool.map(lambda project: retrieve_project(project, ids[project], cfg), sorted(ids)))
    coverage = {}
    for filename in ("messages.csv", "missing.csv"):
        temp = output / (filename + ".tmp")
        with temp.open("w", encoding="utf-8", newline="") as stream:
            columns = ["project", "commit_id", "message", "source"] if filename == "messages.csv" else ["project", "commit_id"]
            writer = csv.writer(stream)
            writer.writerow(columns)
            for row in tables["total"].values():
                project, sha = row["project"], row["commit_id"]
                messages, source, error = results[project]
                if sha not in messages and project in HADOOP_HISTORY:
                    messages, source, _ = results["apache/hadoop"]
                found = sha in messages
                if filename == "messages.csv":
                    stats = coverage.setdefault(project, {"found": 0, "missing": 0, "error": error})
                    stats["found" if found else "missing"] += 1
                    if found:
                        writer.writerow([project, sha, messages[sha], source])
                elif not found:
                    writer.writerow([project, sha])
        os.replace(temp, output / filename)
    write_json(output / "coverage.json", coverage)
    print(json.dumps({key: sum(row[key] for row in coverage.values()) for key in ("found", "missing")}))


if __name__ == "__main__":
    main()
