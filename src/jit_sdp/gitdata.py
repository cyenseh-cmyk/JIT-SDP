"""Explicit Git acquisition, bounded retries, and canonical patch parsing."""
import re
import json
import sqlite3
import subprocess
import time
from pathlib import Path
from .data import audit_archive, key
from .io import read_json, write_json

CODE_EXTENSIONS = {".java", ".groovy", ".scala", ".kt", ".kts", ".py", ".c", ".h", ".cpp", ".hpp", ".js", ".ts", ".sh"}


def git(repo, *args):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=180,
                            encoding="utf-8", errors="replace")
    if result.returncode:
        raise RuntimeError(result.stderr.strip()[-1000:])
    return result.stdout


def parse_patch(text):
    files, file, hunk = [], None, None
    expected = None

    def finish():
        if hunk is not None and expected is not None:
            old = sum(x["kind"] != "+" for x in hunk["lines"])
            new = sum(x["kind"] != "-" for x in hunk["lines"])
            if (old, new) != expected:
                raise ValueError("Incomplete or malformed hunk")

    for line in text.split("\n"):
        if line.startswith("diff --git "):
            finish()
            file = {"old_path": None, "new_path": None, "hunks": [], "binary": False}
            files.append(file)
            hunk, expected = None, None
        elif file is not None:
            if line.startswith("@@ "):
                finish()
                match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", line)
                if not match:
                    raise ValueError("Invalid hunk header")
                expected = (int(match[2] or 1), int(match[4] or 1))
                hunk = {"header": line, "lines": []}
                file["hunks"].append(hunk)
            elif hunk is not None and line[:1] in ("+", "-", " "):
                hunk["lines"].append({"kind": line[0], "text": line[1:]})
            elif hunk is None and line.startswith(("--- ", "+++ ")):
                path = line[4:]
                # Git's quotePath=false leaves ordinary Unicode paths readable.
                if path.startswith('"'):
                    import ast
                    path = ast.literal_eval(path)
                file["old_path" if line.startswith("---") else "new_path"] = (
                    path[2:] if path.startswith(("a/", "b/")) else path)
            elif line.startswith(("Binary files ", "GIT binary patch")):
                file["binary"] = True
    finish()
    return files


def retrieve(repo, project, sha, raw_path):
    fields = git(repo, "show", "-s", "--format=%H%n%P%n%at%n%ct%n%B", sha).split("\n", 4)
    if fields[0] != sha:
        raise ValueError("Git returned a different SHA")
    parents = fields[1].split()
    if len(parents) != 1:
        raise ValueError("root_commit" if not parents else "merge_commit")
    patch = git(repo, "-c", "core.quotePath=false", "diff", "--no-ext-diff", "--no-textconv",
                "--no-renames", "--unified=3", parents[0], sha, "--")
    return cache_record(project, sha, raw_path, fields[1:], patch)


def cache_record(project, sha, raw_path, metadata, patch, *, write_patch=True):
    parent_text, author_time, committer_time, message = metadata
    parents = parent_text.split()
    if len(parents) != 1:
        raise ValueError("root_commit" if not parents else "merge_commit")
    if write_patch:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_text(patch, encoding="utf-8")
    all_files = parse_patch(patch)
    files = [f for f in all_files if not f["binary"] and f["hunks"] and
             Path(f["new_path"] if f["new_path"] != "/dev/null" else f["old_path"]).suffix in CODE_EXTENSIONS]
    if not files or not any(l["kind"] in ("+", "-") for f in files for h in f["hunks"] for l in h["lines"]):
        raise ValueError("no_usable_code_change")
    return {"project": project, "commit_id": sha, "parent_ids": parents,
            "message": message.rstrip("\n"), "retrieved_author_time": int(author_time),
            "committer_time": int(committer_time), "files": files,
            "all_file_count": len(all_files), "filtered_file_count": len(all_files) - len(files),
            "raw_patch_path": str(raw_path) if write_patch else None,
            "retrieval_status": "ok", "parse_status": "ok", "parser_version": 1}


def read_patch_batch(repo, ids):
    """Read parent-relative patches and commit metadata in one Git process."""
    process = subprocess.run(
        ["git", "-C", str(repo), "-c", "core.quotePath=false", "diff-tree", "--stdin",
         "--always", "-r", "-p", "--no-ext-diff", "--no-textconv", "--no-renames",
         "--unified=3", "--format=%x00%H%x00%P%x00%at%x00%ct%x00%B%x00"],
        input="\n".join(ids) + "\n", capture_output=True, encoding="utf-8",
        errors="replace", timeout=600)
    parts = re.split(r"\x00([0-9a-f]{40})\x00", process.stdout)
    records = {}
    for sha, body in zip(parts[1::2], parts[2::2]):
        fields = body.split("\x00", 4)
        if len(fields) == 5:
            records[sha] = (fields[:4], fields[4].lstrip("\n"))
    return records, process.stderr.strip()[-1000:]


def enrich_batches(cfg, rows, retry_errors):
    from collections import Counter, defaultdict
    from concurrent.futures import ThreadPoolExecutor
    from contextlib import closing

    work_dir = Path(cfg["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    database = work_dir / "enrichment.sqlite3"
    with closing(sqlite3.connect(database)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS enrichment ("
                   "project TEXT, sha TEXT, record TEXT, patch TEXT, PRIMARY KEY(project, sha))")
        sql = "SELECT project, sha, record FROM enrichment" if retry_errors else "SELECT project, sha, NULL FROM enrichment"
        completed = {(p, sha) for p, sha, record in db.execute(sql)
                     if not retry_errors or json.loads(record)["retrieval_status"] == "ok"}
    projects = defaultdict(list)
    for row in sorted(rows.values(), key=key):
        project, sha = row["project"], row["commit_id"]
        if cfg["projects"] and project not in cfg["projects"]:
            continue
        if (project, sha) in completed:
            continue
        cache = work_dir / "cache" / project / (sha + ".json")
        if cache.exists():
            saved = read_json(cache)
            if saved.get("parser_version") == 1 and (saved.get("retrieval_status") == "ok" or not retry_errors):
                continue
        projects[project].append(sha)

    def run_project(item):
        project, ids = item
        with closing(sqlite3.connect(database, timeout=60)) as db:
            return enrich_project(cfg, work_dir, project, ids, db)

    counts = Counter()
    with ThreadPoolExecutor(max_workers=4) as workers:
        for result in workers.map(run_project, projects.items()):
            counts.update(result)
    write_json(work_dir / "retrieval_summary.json", {"processed": sum(counts.values()), "status": dict(counts)})
    return dict(counts)


def enrich_project(cfg, work_dir, project, ids, db):
    counts, processed = {}, 0
    hadoop = ("apache/hadoop", "apache/hadoop-hdfs", "apache/hadoop-mapreduce")
    repositories = cfg["repositories"]
    default = work_dir / "repositories" / project.replace("/", "__")
    repo = repositories.get(project, default)
    fallbacks = [repositories[name] for name in hadoop
                 if project in hadoop and name != project and name in repositories]
    batch_size = 500
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        found, error = read_patch_batch(repo, batch)
        for alternate in fallbacks:
            missing = [sha for sha in batch if sha not in found]
            if not missing:
                break
            additional, _ = read_patch_batch(alternate, missing)
            found.update(additional)
        entries = []
        for sha in batch:
            raw = work_dir / "raw" / project / (sha + ".patch")
            patch = None
            try:
                if sha in found:
                    metadata, patch = found[sha]
                    result = cache_record(project, sha, raw, metadata, patch, write_patch=False)
                else:
                    result = retrieve(repo, project, sha, raw)
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                result = {"project": project, "commit_id": sha, "retrieval_status": "error",
                          "parse_status": "error", "reason": str(exc) or error, "parser_version": 1}
            entries.append((project, sha, json.dumps(result), patch))
            status = result["retrieval_status"]
            counts[status] = counts.get(status, 0) + 1
            processed += 1
        with db:
            db.executemany("INSERT OR REPLACE INTO enrichment VALUES (?, ?, ?, ?)", entries)
        print(f"{project}: {processed}/{len(ids)} commits processed in this run, {counts}", flush=True)
    return counts


def acquire(cfg):
    """Clone only selected Apache sources. Hadoop identifiers remain explicitly mapped."""
    tables = audit_archive(cfg)
    projects = sorted({r["project"] for r in tables["total"].values()})
    root = Path(cfg["work_dir"]) / "repositories"
    root.mkdir(parents=True, exist_ok=True)
    for project in projects:
        if cfg["projects"] and project not in cfg["projects"]:
            continue
        if project in cfg["repositories"]:
            git(cfg["repositories"][project], "rev-parse", "--git-dir")
            continue
        if "hadoop" in project.lower():
            raise ValueError(f"Map historical Hadoop source explicitly in repositories: {project}")
        dest = root / project.replace("/", "__")
        if dest.exists():
            git(dest, "rev-parse", "--git-dir")
            continue
        subprocess.run(["git", "clone", "--mirror", "https://github.com/" + project + ".git", str(dest)],
                       check=True, timeout=7200)


def enrich(cfg, limit=None, retry_errors=False):
    tables = audit_archive(cfg)
    rows = {**tables["train"], **tables["test_large"]}
    if limit is None and cfg["repositories"]:
        return enrich_batches(cfg, rows, retry_errors)
    processed = 0
    counts = {}
    for ident, row in sorted(rows.items()):
        project, sha = row["project"], row["commit_id"]
        if cfg["projects"] and project not in cfg["projects"]:
            continue
        cache = Path(cfg["work_dir"]) / "cache" / project / (sha + ".json")
        if cache.exists():
            saved = read_json(cache)
            if saved.get("parser_version") == 1 and (saved.get("retrieval_status") == "ok" or not retry_errors):
                continue
        if limit is not None and processed >= limit:
            break
        repo = cfg["repositories"].get(project)
        if not repo:
            if "hadoop" in project.lower():
                raise ValueError(f"Explicit repository mapping required for {project}")
            repo = str(Path(cfg["work_dir"]) / "repositories" / project.replace("/", "__"))
        raw = Path(cfg["work_dir"]) / "raw" / project / (sha + ".patch")
        for attempt in range(3):
            try:
                result = retrieve(repo, project, sha, raw)
                break
            except (RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
                result = {"project": project, "commit_id": sha, "retrieval_status": "error",
                          "parse_status": "error", "reason": str(exc), "parser_version": 1}
                if isinstance(exc, ValueError):
                    break
                if attempt < 2:
                    time.sleep(attempt + 1)
        write_json(cache, result)
        status = result["retrieval_status"]
        counts[status] = counts.get(status, 0) + 1
        processed += 1
        if processed % 100 == 0:
            print(f"Retrieved {processed} commits: {counts}", flush=True)
    write_json(Path(cfg["work_dir"]) / "retrieval_summary.json", {"processed": processed, "status": counts})
    return counts
