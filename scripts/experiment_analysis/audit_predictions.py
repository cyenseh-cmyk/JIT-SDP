from pathlib import Path
from collections import Counter,defaultdict
import json,csv,io,zipfile,re,hashlib
import numpy as np
from sklearn.metrics import f1_score,matthews_corrcoef,roc_auc_score
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("project", type=Path, help="Restored project containing data, reports and runs")
parser.add_argument("output", type=Path, help="Directory for the analysis outputs")
args = parser.parse_args()
root = args.project.resolve()
here = args.output.resolve()
here.mkdir(parents=True, exist_ok=True)
load=lambda p:json.loads(p.read_text())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=True,allow_nan=False).encode()).hexdigest()
def write(name,rows):
    with (here/name).open("x",newline="",encoding="utf8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
s=load(root/"data/manifests/split.json");assert s["hash"]==digest({k:v for k,v in s.items() if k!="hash"})
with zipfile.ZipFile(root/"apachejit_dataset_replication.zip") as z:
    published={}
    for part in ["train","test_large"]:
        with z.open("apachejit/dataset/apachejit_"+part+".csv") as f:
            published[part]={(r["project"],r["commit_id"]):r["buggy"].lower()=="true" for r in csv.DictReader(io.TextIOWrapper(f))}
keys=[];test_y=[];meta=[];all_ids=set()
for part in ["train","validation","test"]:
    data=s["partitions"][part];pair=[];ys=[];ts=[];h=hashlib.sha256();h.update(b"[")
    with (root/"data/manifests"/(part+".jsonl")).open(encoding="utf8") as f:
        for i,line in enumerate(f):
            rec=json.loads(line);k=(rec["project"],rec["commit_id"]);assert re.fullmatch("[0-9a-f]{40}",k[1])
            assert k not in all_ids and rec["buggy"]==published["test_large" if part=="test" else "train"][k];all_ids.add(k)
            pair.append(k);ys.append(int(rec["buggy"]));ts.append(rec["committer_time"])
            if i:h.update(b", ")
            h.update(json.dumps(rec,sort_keys=True,allow_nan=False).encode())
            if part=="test":
                counts=[Counter(l["kind"] for l in hunk["lines"]) for file in rec["files"] for hunk in file["hunks"]]
                adds=sum(c["+"] for c in counts);dels=sum(c["-"] for c in counts)
                meta.append((bool(adds+dels),any(c["+"] and c["-"] for c in counts),bool(rec["filtered_file_count"] or adds!=rec["la"] or dels!=rec["ld"])))
    h.update(b"]")
    assert h.hexdigest()==data["content_hash"] and sorted(":".join(k) for k in pair)==sorted(data["keys"])
    assert (len(pair),sum(ys),min(ts),max(ts))==(data["count"],data["buggy"],data["time_min"],data["time_max"])
    if part=="test":keys=pair;test_y=np.array(ys)
parts=s["partitions"];assert parts["train"]["time_max"]<parts["validation"]["time_min"]<=parts["validation"]["time_max"]<parts["test"]["time_min"]
report=load(root/"reports/experiment_v5_v6/comparison.json")
look={(r["model"],r["seed"],r["project"]):r for r in report["performance"]}
ids={k:i for i,k in enumerate(keys)};projects=np.array([p for p,c in keys]);n=len(keys)
models=["deepjit","cc2vec","jitline_apachejit"]; seeds=[11,23,37];conditions=["original","T1","T2","T3","T4"]
scores=np.empty((3,3,n));decisions=np.full((3,3,5,n),-9,dtype=np.int8)
exp=[];stress=[];reasons=Counter();odds=Counter()
for mi,model in enumerate(models):
 run="experiment_v5" if mi==0 else "experiment_v6"
 identity=load(root/"runs"/run/"experiment.json");assert identity["split_hash"]==s["hash"]
 for si,seed in enumerate(seeds):
    d=root/"runs"/run/model/f"seed_{seed}";t=load(d/"threshold.json");assert t["partition"]=="validation" and t["metrics"]["n"]==parts["validation"]["count"]
    thr=t["threshold"];assert load(d/"complete.json")["identity"]==digest(identity)
    base={};counts=defaultdict(Counter);hash1={};status2={};flags={}
    with (d/"predictions.csv").open(newline="",encoding="utf8") as f:
        for r in csv.DictReader(f):
            proj,cid=r["project"],r["commit_id"];i=ids[(proj,cid)];cond=r["condition"];ci=conditions.index(cond)
            assert decisions[mi,si,ci,i]==-9
            assert (r["model"],int(r["seed"]),r["run_id"],r["partition"],float(r["threshold"]),int(r["y_original"]))==(model,seed,run,"test",thr,test_y[i])
            p=float(r["p_original"]);y=int(r["yhat_original"]);hs=r["input_hash_original"]
            assert 0<=p<=1 and int(p>=thr)==y and re.fullmatch("[0-9a-f]{64}",hs)
            sig=(p,y,hs)
            if ci==0: base[i]=sig;scores[mi,si,i]=p
            else:assert base[i]==sig
            changes,two,ms=meta[i]
            reason=("no_message_channel" if cond=="T4" and mi!=0 else "no_changed_lines" if ci in (1,2,3) and not changes else "no_two_sided_hunk" if cond=="T2" and not two else "metric_scope_mismatch" if mi==2 and ci in (1,2) and ms else "")
            assert reason==r["status_reason"] and (r["metric_scope_ok"]=="True")==(reason!="metric_scope_mismatch")
            status=r["status"];hh=r["input_hash_transformed"]
            if reason:
                assert status=="structural_na" and not (hh or r["p_transformed"] or r["yhat_transformed"])
                pred=-1;reasons[reason]+=1
            else:
                pp=float(r["p_transformed"]);pred=int(r["yhat_transformed"])
                assert 0<=pp<=1 and int(pp>=thr)==pred and re.fullmatch("[a-f0-9]{64}",hh)
                assert status==("original" if not ci else "encoding_noop" if hh==hs else "effective_change")
                if not ci:assert pp==p and hh==hs
                if hh==hs and (abs(p-pp)>1e-5 or y!=pred):odds[(model,cond)]+=1
            if ci==1:hash1[i]=hh;flags[i]=r["t1_t2_same_encoding"]
            if ci==2:
                flag=str(hash1[i]==hh) if hash1[i] and hh else ""
                assert flags[i]==r["t1_t2_same_encoding"]==flag
            decisions[mi,si,ci,i]=pred;counts[(proj,cond)][status]+=1
    assert (decisions[mi,si]!=-9).all()
    for proj in sorted(set(projects)|{"pooled"}):
        mask=(projects==proj) if proj!="pooled" else np.ones(n,bool);yy=test_y[mask];pred=decisions[mi,si,0,mask];p=scores[mi,si,mask]
        two=len(set(yy))==2;met={"mcc":float(matthews_corrcoef(yy,pred)) if two else None,"f1":float(f1_score(yy,pred,zero_division=0)),"roc_auc":float(roc_auc_score(yy,p)) if two else None}
        for k,v in met.items():assert (v is None and look[model,seed,proj][k] is None) or abs(v-look[model,seed,proj][k])<1e-9
        if proj=="pooled":continue
        exp.append(dict(status="READY",run=run,model=model,seed=seed,project=proj,commits=int(mask.sum()),positive=int(yy.sum()),threshold=thr,**met))
        for cond in conditions[1:]:
            c=counts[proj,cond];app=c["effective_change"]+c["encoding_noop"]
            stress.append(dict(status="READY" if app else "NOT APPLICABLE",model=model,seed=seed,project=proj,condition=cond,expected=int(mask.sum()),applicable=app,effective=c["effective_change"],noop=c["encoding_noop"],structural_na=c["structural_na"],missing=0))
    print(model,seed,"OK",flush=True)
assert np.all((decisions[:,:,1:]>=0)==(decisions[:,0:1,1:]>=0))
write("experiment-matrix.csv",exp);write("counterfactual-matrix.csv",stress)
np.savez_compressed(here/"audit-inputs.npz",y=test_y,p=scores,d=decisions,projects=projects)
(here/"coverage-check.json").write_text(json.dumps({"test":n,"projects":sorted(set(projects)),"rows_checked":int(decisions.size),"na_reasons":dict(reasons),"same_hash_score_or_decision_exceptions":{str(k):v for k,v in odds.items()},"split_hash":s["hash"]},indent=2))
