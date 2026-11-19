from copy import deepcopy
from .features import changed_lines

CONDITIONS = ("T1", "T2", "T3", "T4")


def transform(record, condition, model):
    result = deepcopy(record)
    if condition not in CONDITIONS:
        raise ValueError(condition)
    if condition == "T4":
        if model in ("cc2vec", "jitline_apachejit"):
            return result, "no_message_channel", 0
        result["message"] = ""
        return result, "", 0
    lines = changed_lines(record)
    if not lines:
        return result, "no_changed_lines", 0
    changed = 0
    delta_added = 0
    for file in result["files"]:
        for hunk in file["hunks"]:
            deleted = [l["text"] for l in hunk["lines"] if l["kind"] == "-"]
            added = [l["text"] for l in hunk["lines"] if l["kind"] == "+"]
            if condition == "T1":
                for line in hunk["lines"]:
                    if line["kind"] in ("+", "-"):
                        line["kind"] = "-" if line["kind"] == "+" else "+"
                delta_added += len(deleted) - len(added)
                changed += bool(deleted or added)
            elif condition == "T2" and deleted and added:
                # Swap old/new blocks in each edit region, retaining intervening context.
                # Empty region sides are exchanged too, but wholly one-sided hunks
                # remain unchanged. This is an involution even for unequal blocks.
                new_lines, region = [], []
                def flush():
                    old = [l["text"] for l in region if l["kind"] == "-"]
                    new = [l["text"] for l in region if l["kind"] == "+"]
                    new_lines.extend({"kind": "-", "text": text} for text in new)
                    new_lines.extend({"kind": "+", "text": text} for text in old)
                    region.clear()
                for line in hunk["lines"]:
                    if line["kind"] in ("-", "+"):
                        region.append(line)
                    else:
                        flush()
                        new_lines.append(line)
                flush()
                hunk["lines"] = new_lines
                delta_added += len(deleted) - len(added)
                changed += 1
    if condition == "T2" and not changed:
        return result, "no_two_sided_hunk", 0
    if condition == "T3":
        result["synthetic_prefix"] = "<ADD> <DEL> <ADD> <DEL>"
    if model == "jitline_apachejit" and condition in ("T1", "T2"):
        la = sum(l["kind"] == "+" for l in lines)
        ld = sum(l["kind"] == "-" for l in lines)
        if record.get("filtered_file_count", 0) or la != record["la"] or ld != record["ld"]:
            return result, "metric_scope_mismatch", changed
        result["la"] += delta_added
        result["ld"] -= delta_added
    return result, "", changed
