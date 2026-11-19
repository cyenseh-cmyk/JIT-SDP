"""Upstream architectures with explicit CPU/batch and training corrections."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def upstream(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load upstream source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def classifier(cfg, state, name):
    if name == "cc2vec":
        from torch import nn
        return nn.Sequential(nn.Linear(3 * cfg["cc2vec"]["embed_size"] + 4, 1),
                             nn.Flatten(start_dim=0), nn.Sigmoid())
    if name != "deepjit":
        raise ValueError(f"Unsupported neural model: {name}")
    base = Path(cfg["baseline_dir"])
    args = SimpleNamespace(vocab_msg=len(state["message"]), vocab_code=len(state["code"]),
        embedding_dim=cfg["neural"]["embedding_dim"], class_num=1,
        num_filters=cfg["neural"]["num_filters"], filter_sizes=cfg["neural"]["filter_sizes"],
        hidden_units=cfg["neural"]["hidden_units"], dropout_keep_prob=cfg["neural"]["dropout"])
    cls = upstream(base / "DeepJIT/model.py", "jit_sdp_upstream_deepjit").DeepJIT
    if cfg.get("deepjit", {}).get("cuda_backward") == "triton":
        import torch
        device = torch.device(cfg["neural"]["device"])
        if device.type == "cuda":
            if torch.version.cuda is None or torch.cuda.get_device_capability(device)[0] < 8:
                raise ValueError("DeepJIT Triton backward requires an NVIDIA SM80 or newer GPU")
            from .deepjit_cuda import accelerated_model
            return accelerated_model(cls, args)
    return cls(args)


def representation(cfg, state):
    import torch
    ref = upstream(Path(cfg["baseline_dir"]) / "CC2Vec/jit_cc2ftr_model.py", "jit_sdp_upstream_cc2vec")

    class PortableWord(ref.WordRNN):
        def forward(self, inp, hid_state):
            output, hidden = self.wordRNN(self.embed(inp), hid_state)
            weights = torch.softmax(self.attn_combine(self.wordattn(output)), dim=0)
            return (output * weights).sum(dim=0, keepdim=True), hidden

    class PortableSentence(ref.SentRNN):
        def forward(self, inp, hid_state):
            output, hidden = self.sentRNN(inp, hid_state)
            weights = torch.softmax(self.attn_combine(self.sentattn(output)), dim=0)
            return (output * weights).sum(dim=0, keepdim=True), hidden

    class PortableCC2Vec(ref.HierachicalRNN):
        def __init__(self, args):
            super().__init__(args)
            self.wordRNN = PortableWord(self.vocab_size, self.embed_size, self.batch_size, self.hidden_size)
            self.sentRNN = PortableSentence(self.embed_size, self.hidden_size)

        def forward_code(self, x, hid_state):
            # Correct the reference's batch/word reshape and CUDA-only conversion.
            # Normalize attention over sequence positions, never other batch items.
            # Keep its comparison operators and mean over retained file slots.
            _, sent_state, word_state = hid_state
            files = []
            for fidx in range(x.shape[1]):
                sentences = []
                for lidx in range(x.shape[2]):
                    words = x[:, fidx, lidx, :].transpose(0, 1).contiguous()
                    sentence, _ = self.wordRNN(words, word_state)
                    sentences.append(sentence)
                file, _ = self.sentRNN(torch.cat(sentences, dim=0), sent_state)
                files.append(file)
            return torch.cat(files, dim=0).mean(dim=0)

        def states(self, batch):
            self.batch_size = batch
            self.wordRNN.batch_size = batch
            zero = next(self.parameters()).new_zeros(2, batch, self.hidden_size)
            return zero, zero.clone(), zero.clone()

        def logits(self, added, removed):
            feature = self.forward_commit_embeds_diff(added, removed, *self.states(len(added)))
            return self.fc2(torch.relu(self.fc1(self.dropout(feature))))

        def features(self, added, removed):
            return self.forward_commit_embeds_diff(added, removed, *self.states(len(added)))

    args = SimpleNamespace(vocab_code=len(state["code"]), batch_size=cfg["cc2vec"]["batch_size"],
        embed_size=cfg["cc2vec"]["embed_size"], hidden_size=cfg["cc2vec"]["hidden_size"],
        class_num=len(state["message"]), dropout_keep_prob=cfg["neural"]["dropout"])
    return PortableCC2Vec(args)


def seed_everything(seed, threads):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


def batch_input(adapter, records, device, names=("message", "code", "added", "removed")):
    import numpy as np
    import torch
    values = [adapter.encode(r, names) for r in records]
    return {name: torch.as_tensor(np.stack([r[name] for r in values]), device=device)
            for name in names}


class NeuralPipeline:
    def __init__(self, cfg, adapter, model_name, seed):
        import torch
        self.cfg, self.adapter, self.name, self.seed = cfg, adapter, model_name, seed
        self.device = torch.device(cfg["neural"]["device"])
        self.encoder = None
        self.model = None
        self.history = {}
        self.validation_scores = None

    def _forward(self, batch, feature=None):
        if self.encoder is None:
            return self.model(batch["message"], batch["code"])
        if feature is None:
            import torch
            with torch.no_grad():
                feature = self.encoder.features(batch["added"], batch["removed"])
        return self.model(feature)

    def _batch(self, records, auxiliary=False, feature=None):
        if feature is not None:
            return {}
        names = self.adapter.input_names(self.name)
        if auxiliary:
            names += ("message",)
        return batch_input(self.adapter, records, self.device, names)

    def _cache_features(self, records):
        import torch

        size = self.cfg["cc2vec"]["batch_size"]
        with torch.no_grad():
            chunks = []
            for start in range(0, len(records), size):
                batch = self._batch(records[start:start + size])
                chunks.append(self.encoder.features(batch["added"], batch["removed"]).cpu())
            return torch.cat(chunks)

    def _fit_stage(self, model, train, validation, settings, auxiliary, checkpoint,
                   train_features=None, validation_features=None):
        import numpy as np
        import torch
        from tqdm.auto import tqdm
        from .metrics import classification
        optimizer = torch.optim.Adam(model.parameters(), lr=self.cfg["neural"]["learning_rate"],
                                     weight_decay=self.cfg["neural"]["weight_decay"])
        criterion = torch.nn.BCEWithLogitsLoss() if auxiliary else torch.nn.BCELoss()
        generator = np.random.default_rng(self.seed)
        best, patience_best, stale, history = None, None, 0, []
        stage = "representation" if auxiliary else "classifier"
        early_stopping = self.name != "deepjit" or self.cfg.get("deepjit", {}).get("early_stopping", True)
        for epoch in range(1, settings["max_epochs"] + 1):
            label = f"{self.name} seed {self.seed} {stage} epoch {epoch}/{settings['max_epochs']}"
            model.train()
            total = 0.0
            indices = generator.permutation(len(train))
            for start in tqdm(range(0, len(train), settings["batch_size"]),
                              desc=f"{label} train", unit="batch", dynamic_ncols=True):
                batch_indices = indices[start:start + settings["batch_size"]]
                records = [train[i] for i in batch_indices]
                feature = None if train_features is None else train_features[batch_indices].to(self.device)
                batch = self._batch(records, auxiliary, feature)
                optimizer.zero_grad(set_to_none=True)
                if auxiliary:
                    prediction = model.logits(batch["added"], batch["removed"])
                    target = torch.zeros((len(records), len(self.adapter.state["message"])), device=self.device)
                    target.scatter_(1, batch["message"], 1.0)
                    # Padding and unknown are not real auxiliary message terms.
                    loss = criterion(prediction[:, 2:], target[:, 2:])
                else:
                    prediction = self._forward(batch, feature)
                    target = torch.tensor([r["buggy"] for r in records], dtype=torch.float32, device=self.device)
                    loss = criterion(prediction, target)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite training loss")
                loss.backward()
                optimizer.step()
                total += loss.item() * len(records)
            model.eval()
            probabilities, validation_loss = [], 0.0
            with torch.no_grad():
                for start in tqdm(range(0, len(validation), settings["batch_size"]),
                                  desc=f"{label} validation", unit="batch", dynamic_ncols=True):
                    records = validation[start:start + settings["batch_size"]]
                    feature = None if validation_features is None else validation_features[
                        start:start + len(records)].to(self.device)
                    batch = self._batch(records, auxiliary, feature)
                    if auxiliary:
                        pred = model.logits(batch["added"], batch["removed"])
                        target = torch.zeros_like(pred)
                        target.scatter_(1, batch["message"], 1.0)
                        validation_loss += criterion(pred[:, 2:], target[:, 2:]).item() * len(records)
                    else:
                        probabilities.extend(self._forward(batch, feature).cpu().tolist())
            score = (-validation_loss / len(validation)) if auxiliary else classification(
                [r["buggy"] for r in validation], probabilities)["mcc"]
            if score is None or not np.isfinite(score):
                raise ValueError("Undefined validation checkpoint objective")
            history.append({"epoch": epoch, "train_loss": total / len(train),
                            "validation_auxiliary_loss" if auxiliary else "validation_mcc_at_0_5": -score if auxiliary else score})
            print(f"{label}: train_loss={total / len(train):.5f}, selection={score:.5f}", flush=True)
            if best is None or score > best:
                best = score
                torch.save(model.state_dict(), checkpoint)
            if patience_best is None or score > patience_best + self.cfg["neural"]["min_delta"]:
                patience_best, stale = score, 0
            else:
                stale += 1
            if early_stopping and stale >= settings["patience"]:
                break
        model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        model.eval()
        return history

    def fit(self, train, validation, out):
        seed_everything(self.seed, self.cfg["neural"]["threads"])
        train_features = validation_features = None
        if self.name == "cc2vec":
            if len(self.adapter.state["message"]) <= 2:
                raise ValueError("No message terms in training data for auxiliary learning")
            self.encoder = representation(self.cfg, self.adapter.state).to(self.device)
            self.history["representation"] = self._fit_stage(self.encoder, train, validation,
                self.cfg["cc2vec"], True, out / "encoder.pt")
            self.encoder.requires_grad_(False)
            self.encoder.eval()
            train_features = self._cache_features(train)
            validation_features = self._cache_features(validation)
        self.model = classifier(self.cfg, self.adapter.state, self.name).to(self.device)
        self.history["classifier"] = self._fit_stage(self.model, train, validation,
            self.cfg["neural"], False, out / "classifier.pt", train_features, validation_features)
        self.validation_scores = self.predict(validation, features=validation_features)
        return self

    def predict(self, records, *, features=None):
        import torch
        self.model.eval()
        if self.encoder is not None:
            self.encoder.eval()
        result = []
        with torch.no_grad():
            size = self.cfg["cc2vec" if self.encoder is not None else "neural"]["batch_size"]
            for start in range(0, len(records), size):
                chunk = records[start:start + size]
                feature = None if features is None else features[start:start + len(chunk)].to(self.device)
                batch = self._batch(chunk, feature=feature)
                result.extend(self._forward(batch, feature).cpu().tolist())
        return result

    def load(self, out):
        import torch
        self.model = classifier(self.cfg, self.adapter.state, self.name).to(self.device)
        if self.name == "cc2vec":
            self.encoder = representation(self.cfg, self.adapter.state).to(self.device)
            self.encoder.load_state_dict(torch.load(out / "encoder.pt", map_location=self.device, weights_only=True))
            self.encoder.requires_grad_(False).eval()
        self.model.load_state_dict(torch.load(out / "classifier.pt", map_location=self.device, weights_only=True))
        self.model.eval()
        return self
