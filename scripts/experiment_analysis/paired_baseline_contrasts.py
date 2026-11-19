import json,itertools
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
import csv
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("project", type=Path, help="Restored project containing data, reports and runs")
parser.add_argument("output", type=Path, help="Directory for the analysis outputs")
args = parser.parse_args()
root = args.project.resolve()
here = args.output.resolve()
here.mkdir(parents=True, exist_ok=True)
a=np.load(here/"audit-inputs.npz");y=a["y"];projects=a["projects"];p=a["p"];d=a["d"][:,:,0,:]
models=["deepjit","cc2vec","jitline_apachejit"];groups=[np.flatnonzero(projects==g) for g in np.unique(projects)]
ranks=[[np.unique(x,return_inverse=True)[1] for x in model] for model in p]
def means(w):
    pos=w*y;neg=w*(1-y);P=pos.sum();N=neg.sum();tp=(d*pos).sum(-1);fp=(d*neg).sum(-1)
    fn=P-tp;tn=N-fp
    den=np.sqrt((tp+fp).astype(float)*P*N*(tn+fn))
    mcc=np.divide(tp*tn-fp*fn,den,out=np.zeros(den.shape),where=den>0)
    f1=2*tp/(2*tp+fp+fn)
    auc=np.empty_like(mcc)
    for i in range(3):
        for j in range(3):
            rank=ranks[i][j];aa=np.bincount(rank,weights=pos);bb=np.bincount(rank,weights=neg)
            auc[i,j]=np.dot(aa,np.cumsum(bb)-0.5*bb)/(P*N)
    return np.asarray([mcc.mean(1),f1.mean(1),auc.mean(1)])
rng=np.random.default_rng(2026);orig=means(np.ones(len(y),int));samples=[]
for _ in range(1000):
    idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
    samples.append(means(np.bincount(idx,minlength=len(y))))
S=np.array(samples)
with (here/"rq1-paired-baseline-contrasts.csv").open("x",newline="") as f:
    w=csv.writer(f);w.writerow(["first","second","metric","first_minus_second","paired_ci_low","paired_ci_high","draws"])
    for i,j in itertools.combinations(range(3),2):
        for k,metric in enumerate(["mcc","binary_f1","roc_auc"]):
            v=S[:,k,i]-S[:,k,j];w.writerow([models[i],models[j],metric,orig[k,i]-orig[k,j],*np.quantile(v,[.025,.975]),len(v)])
