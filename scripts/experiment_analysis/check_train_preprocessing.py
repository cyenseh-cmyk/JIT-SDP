from pathlib import Path
import numpy as np,json,joblib
from jit_sdp.features import JITLineAdapter,NeuralAdapter
from jit_sdp.io import read_json,read_jsonl
cfg=read_json("runs/experiment_v6/experiment.json")["config"]
train=list(read_jsonl("data/manifests/train.jsonl"))
vocab=NeuralAdapter(cfg["encoding"]).fit(train).state
checks={}
for n,m in [("experiment_v5","deepjit"),("experiment_v6","cc2vec")]:
 for seed in cfg["seeds"]:
    assert vocab==read_json(Path("runs")/n/m/f"seed_{seed}"/"adapter.json")
checks["six_neural_training_vocabularies_match"]=True
counts=JITLineAdapter(cfg["jitline"]["max_tokens"],cfg["jitline"]["min_df"]).fit(train)
for seed in cfg["seeds"]:
    bundle=joblib.load(Path("runs/experiment_v6/jitline_apachejit")/f"seed_{seed}"/"forest.joblib")
    saved=bundle["adapter"]
    assert saved.vectorizer.vocabulary_==counts.vectorizer.vocabulary_
    for key in ["mean_","var_","scale_","n_samples_seen_"]:
        np.testing.assert_array_equal(getattr(saved.scaler,key),getattr(counts.scaler,key))
    del bundle,saved
checks["three_forest_vocabularies_and_scalers_match_training_only"]=True
print(json.dumps(checks))