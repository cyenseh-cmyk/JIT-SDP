"""Training-fitted adapters; padding=0 and unknown=1 are distinct."""
import re
from collections import Counter
from .io import digest
from .data import METRICS

TOKEN = re.compile(r"[A-Za-z_$][\w$]*|\d+(?:\.\d+)?|[^\w\s]")


def tokens(text, lower=False):
    values = TOKEN.findall(text)
    return [x.lower() for x in values] if lower else values


def changed_lines(record):
    return [line for file in record["files"] for hunk in file["hunks"]
            for line in hunk["lines"] if line["kind"] in ("+", "-")]


def vocabulary(texts, maximum):
    counts = Counter(word for text in texts for word in tokens(text, lower=True))
    ordered = sorted(counts, key=lambda word: (-counts[word], word))[:max(0, maximum - 2)]
    return {"<PAD>": 0, "<UNK>": 1, **{word: index + 2 for index, word in enumerate(ordered)}}


class NeuralAdapter:
    def __init__(self, config, state=None):
        self.config = config
        self.state = state

    def fit(self, records):
        self.state = {
            "message": vocabulary((r["message"] for r in records), self.config["max_msg_vocab"]),
            "code": vocabulary((l["text"] for r in records for l in changed_lines(r)), self.config["max_code_vocab"]),
            "encoding": self.config, "version": 1}
        return self

    @staticmethod
    def input_names(model):
        return ("added", "removed") if model == "cc2vec" else ("message", "code")

    def encode(self, record, names=("message", "code", "added", "removed")):
        import numpy as np
        cfg = self.config
        state = self.state
        if state is None:
            raise ValueError("Adapter is not fitted")
        truncated = False

        def sequence(text, vocab, length):
            nonlocal truncated
            words = tokens(text, lower=True)
            truncated |= len(words) > length
            values = [vocab.get(w, 1) for w in words[:length]]
            return values + [0] * (length - len(values))

        encoded = {}
        if "message" in names:
            encoded["message"] = np.asarray(
                sequence(record["message"], state["message"], cfg["msg_length"]), dtype=np.int64)
        prefix = record.get("synthetic_prefix", "")
        if "code" in names:
            # The reference CNN consumes ordered changed lines, without polarity tags.
            texts = [line["text"] for line in changed_lines(record)]
            if texts and prefix:
                texts[0] = prefix + " " + texts[0]
            truncated |= len(texts) > cfg["code_lines"]
            code = np.zeros((cfg["code_lines"], cfg["code_length"]), dtype=np.int64)
            for index, text in enumerate(texts[:cfg["code_lines"]]):
                code[index] = sequence(text, state["code"], cfg["code_length"])
            encoded["code"] = code
        channels = {kind: np.zeros((cfg["cc_files"], cfg["cc_lines"], cfg["cc_length"]), dtype=np.int64)
                    for kind, name in (("+", "added"), ("-", "removed")) if name in names}
        if channels:
            truncated |= len(record["files"]) > cfg["cc_files"]
            for fidx, file in enumerate(record["files"][:cfg["cc_files"]]):
                lines = [line for hunk in file["hunks"] for line in hunk["lines"]
                         if line["kind"] in channels and line["text"].strip()]
                for kind, array in channels.items():
                    channel = [line["text"] for line in lines if line["kind"] == kind]
                    if prefix and lines and kind == lines[0]["kind"]:
                        channel[0] = prefix + " " + channel[0]
                        prefix = ""
                    truncated |= len(channel) > cfg["cc_lines"]
                    for lidx, text in enumerate(channel[:cfg["cc_lines"]]):
                        array[fidx, lidx] = sequence(text, state["code"], cfg["cc_length"])
            encoded.update({name: channels[kind] for kind, name in (("+", "added"), ("-", "removed"))
                            if kind in channels})
        return {**encoded, "truncated": bool(truncated)}

    def input_hash(self, record, model):
        names = self.input_names(model)
        encoded = self.encode(record, names)
        return digest({name: encoded[name].tolist() for name in names})


def preprocess_jitline(text):
    # Mirrors my_util.preprocess_code_line; preserve case and literal conventions.
    for char in "(){}[].:;,":
        text = text.replace(char, " ")
    text = text.replace(" _ ", "_")
    for pattern in (r"``.*``", r"'.*'", r'".*"'):
        text = re.sub(pattern, "<STR>", text)
    return re.sub(r"\d+", "<NUM>", text).strip()


def jitline_text(record):
    channels = []
    for kind in ("+", "-"):
        lines = [preprocess_jitline(l["text"]) for l in changed_lines(record)
                 if l["kind"] == kind and not l["text"].startswith("#")]
        # Reference loader removes duplicate lines within each channel.
        channels.append(" \n ".join(sorted(set(lines))))
    return record.get("synthetic_prefix", "") + " " + " ".join(channels)


class JITLineAdapter:
    def __init__(self, maximum=50000, min_df=3):
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.preprocessing import StandardScaler
        self.vectorizer = CountVectorizer(lowercase=False, max_features=maximum, min_df=min_df)
        self.scaler = StandardScaler()

    def fit(self, records):
        self.vectorizer.fit([jitline_text(r) for r in records])
        self.scaler.fit([[r[m] for m in METRICS] for r in records])
        return self

    def transform(self, records):
        import numpy as np
        from scipy.sparse import csr_matrix, hstack
        text = self.vectorizer.transform([jitline_text(r) for r in records]).astype(np.float32)
        metrics = self.scaler.transform([[r[m] for m in METRICS] for r in records]).astype(np.float32)
        return hstack([text, csr_matrix(metrics)], format="csr")

    def input_hash(self, record, model=None):
        x = self.transform([record])
        x.sum_duplicates()
        x.sort_indices()
        x.eliminate_zeros()
        return digest([x.shape, x.indptr.tolist(), x.indices.tolist(), x.data.tolist()])
