from pathlib import Path
import csv,json,os,sys,gc,platform,traceback
import numpy as np
import torch
from jit_sdp.data import load_partitions
from jit_sdp.io import read_json,digest,file_hash
from jit_sdp.metrics import classification,select_threshold
from jit_sdp.neural import seed_everything
from jit_sdp.runner import load_pipeline,prediction_rows
from jit_sdp.features import JITLineAdapter
from jit_sdp.data import METRICS
def emit(x):print(json.dumps(x,allow_nan=False),flush=True)
runs={"deepjit":"experiment_v5","cc2vec":"experiment_v6","jitline_apachejit":"experiment_v6"}
cfg=read_json("runs/experiment_v6/experiment.json")["config"]
groups,manifest=load_partitions(cfg)
del groups["train"];gc.collect()
test=groups["test"];val=groups["validation"]
key=lambda r:(r["project"],r["commit_id"])
projects=sorted({r["project"] for r in test})
sample=[next(r for r in test if r["project"]==p) for p in projects]
sample_ids=set(map(key,sample))
tidx={key(r):i for i,r in enumerate(test)}
emit({"kind":"split","hash":manifest["hash"],"projects":projects,"parts":{k:v["count"] for k,v in manifest["partitions"].items()},"metrics":METRICS})
for model,n in runs.items():
 for seed in (11,23,37):
    d=Path("runs")/n/model/f"seed_{seed}"
    try:
        e=read_json(d.parent.parent/"experiment.json");cm=read_json(d/"complete.json")
        assert cm["identity"]==digest(e) and all(file_hash(d/f)==h for f,h in cm["artifacts"].items())
        seed_everything(seed,4);pipe,t=load_pipeline(d)
        def predict(rows):
            return np.concatenate([np.asarray(pipe.predict(rows[i:i+1024])) for i in range(0,len(rows),1024)])
        a=predict(test);v=predict(val); y=[int(r["buggy"]) for r in val]
        recorded=read_json(d/"threshold.json")
        t2=select_threshold(y,v,cfg["threshold_grid_step"]);vm=classification(y,v,t)
        metric_errors={k:abs(vm[k]-s) for k,s in recorded["metrics"].items() if type(s) in (int,float)}
        expected=np.full(len(test),np.nan); target={}
        with (d/"predictions.csv").open(newline="") as f:
            for r in csv.DictReader(f):
                if r["condition"]=="original":expected[tidx[key(r)]]=float(r["p_original"])
                if key(r) in sample_ids:target[key(r)+(r["condition"],)]=r
        replay=prediction_rows(e["config"],pipe,sample,model,seed,t,stress=True)
        maxerr=0.;hash_errors=decision_errors=status_errors=0
        for r in replay:
            b=target[key(r)+(r["condition"],)]
            status_errors+=r["status"]!=b["status"] or r["status_reason"]!=b["status_reason"]
            hash_errors+=r["input_hash_original"]!=b["input_hash_original"] or (r["input_hash_transformed"] or "")!=b["input_hash_transformed"]
            if r["p_transformed"] is not None:
                maxerr=max(maxerr,abs(r["p_transformed"]-float(b["p_transformed"])))
                decision_errors+=r["yhat_transformed"]!=int(b["yhat_transformed"])
        ck={}
        for f in d.glob("*.pt"):
            state=torch.load(f,map_location="cpu",weights_only=True)
            ck[f.name]={"fields":len(state),"all_finite_tensors":all(isinstance(z,torch.Tensor) and bool(z.isfinite().all()) for z in state.values()),"optimizer_keys":bool({"param_groups","optimizer","optimizer_state_dict"}.intersection(state))}
        if model=="jitline_apachejit":ck["forest.joblib"]={"trees":len(pipe.model.estimators_),"tokens":len(pipe.adapter.vectorizer.vocabulary_),"scaled_metrics":int(pipe.adapter.scaler.n_features_in_)}
        emit({"kind":"model","model":model,"seed":seed,"threshold":t,"validation_reselected_threshold":t2,"validation_metric_max_error":max(metric_errors.values()),"test_max_abs_error":float(np.max(np.abs(a-expected))),"test_decision_mismatches":int(np.sum((a>=t)!=(expected>=t))),"sample_stress_max_abs_error":maxerr,"sample_hash_errors":hash_errors,"sample_decision_errors":decision_errors,"sample_status_errors":status_errors,"sample_rows":len(replay),"checkpoints":ck})
        del pipe;gc.collect();torch.cuda.empty_cache()
    except Exception:
        emit({"kind":"error","model":model,"seed":seed,"trace":traceback.format_exc()})
