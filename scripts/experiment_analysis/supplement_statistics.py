"""Audit supplement. Only saved prediction rows are used. Run audit_predictions.py first."""
from pathlib import Path
from collections import Counter
from itertools import permutations,combinations
import csv,json
import numpy as np
from scipy.stats import rankdata,friedmanchisquare
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("project", type=Path, help="Restored project containing data, reports and runs")
parser.add_argument("output", type=Path, help="Directory for the analysis outputs")
args = parser.parse_args()
root = args.project.resolve()
here = args.output.resolve()
here.mkdir(parents=True, exist_ok=True)
a=np.load(here/"audit-inputs.npz");y=a["y"];p=a["p"];d=a["d"];proj=a["projects"]
models=["deepjit","cc2vec","jitline_apachejit"]; seeds=[11,23,37]
rng=np.random.default_rng(2026)
R=json.loads((root/"reports/experiment_v5_v6/comparison.json").read_text())
def out(name,rows):
    with (here/name).open("x",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metrics(tp,fp,P,N):
    tp=np.asarray(tp,float);fp=np.asarray(fp,float); tn=N-fp; fn=P-tp
    den=np.sqrt(np.maximum(0,(tp+fp)*P*N*(tn+fn)))
    m=np.divide(tp*tn-fp*fn,den,out=np.zeros_like(tp),where=den>0)
    if np.isscalar(P) and (P==0 or N==0):m=np.full_like(tp,np.nan)
    f=np.divide(2*tp,2*tp+fp+fn,out=np.zeros_like(tp),where=(2*tp+fp+fn)>0)
    return m,f
def ci(samples):
    v=np.asarray(samples);v=v[np.isfinite(v)]
    return dict(lo=float(np.percentile(v,2.5)) if len(v) else None,hi=float(np.percentile(v,97.5)) if len(v) else None,retained=len(v),undefined=1000-len(v),unstable=len(v)<900)
def holm(rows,field="p"):
    eligible=sorted([r for r in rows if r[field] is not None],key=lambda r:r[field]);prev=0
    for i,r in enumerate(eligible):
        prev=max(prev,min(1,r[field]*(len(eligible)-i)));r["p_holm"]=prev
def draws(P):
    groups=[np.flatnonzero(P==g) for g in np.unique(P)]
    for _ in range(1000):
        idx=np.concatenate([rng.choice(g,len(g),replace=True) for g in groups])
        yield np.bincount(idx,minlength=len(P))
def stats(scores,dec,Y,w,rank):
    pos=w*Y;neg=w*(1-Y);P=pos.sum();N=neg.sum()
    tp=(dec*pos).sum();fp=(dec*neg).sum();m,f=metrics(tp,fp,P,N)
    pp=np.bincount(rank,weights=pos);nn=np.bincount(rank,weights=neg,minlength=len(pp))
    auc=np.dot(pp,np.cumsum(nn)-.5*nn)/(P*N) if P*N else np.nan
    return [float(m),float(f),float(auc)]
rq1=[]
for project in ["pooled",*np.unique(proj)]:
    sel=np.ones(len(y),bool) if project=="pooled" else proj==project
    Y=y[sel]; P=proj[sel];X=p[:,:,sel];D=d[:,:,0,:][:,:,sel]
    ranks=[[np.unique(X[i,j],return_inverse=True)[1] for j in range(3)] for i in range(3)]
    def allmetrics(w):
        return np.array([[stats(X[i,j],D[i,j],Y,w,ranks[i][j]) for j in range(3)] for i in range(3)])
    orig=allmetrics(np.ones(len(Y),int))
    samples=np.array([allmetrics(w).mean(axis=1) for w in draws(P)])
    for i,m in enumerate(models):
        for j,metric in enumerate(["mcc","binary_f1","roc_auc"]):
            v=orig[i,:,j];row=dict(project=project,model=m,metric=metric,n=len(Y),mean=v.mean(),sample_sd=v.std(ddof=1))
            rq1.append(dict(**row,**ci(samples[:,i,j])))
out("rq1-fixed-project-bootstrap.csv",rq1)
def confusion_scores(Y,D,w):
    return metrics((D*(Y*w)).sum(-1),(D*((1-Y)*w)).sum(-1),(Y*w).sum(),((1-Y)*w).sum())[0]
rq2=[];projectrq2=[]
for i,model in enumerate(models):
 for t in range(1,5):
    ok=d[i,0,t]>=0
    if not ok.any():continue
    for project in ["pooled",*np.unique(proj)]:
        sel=ok if project=="pooled" else (ok & (proj==project))
        if not sel.any():continue
        Y=y[sel];A=d[i,:,0,:][:,sel];B=d[i,:,t,:][:,sel];P=proj[sel];W=np.ones(len(Y),int)
        observed=(confusion_scores(Y,A,W)-confusion_scores(Y,B,W)).mean()
        if len(set(Y))<2:
            projectrq2.append(dict(project=project,model=model,condition="T"+str(t),n=len(Y),delta_mcc=None,**ci([])));continue
        # Group identical label/decision patterns. Binomial counts represent independent
        # within-commit swaps, with the same swap across the three seeds.
        pat,ct=np.unique(np.column_stack([Y,A.T,B.T]),axis=0,return_counts=True)
        yy=pat[:,0];aa=pat[:,1:4];bb=pat[:,4:7];diff=bb-aa
        samples=[(confusion_scores(Y,A,w)-confusion_scores(Y,B,w)).mean() for w in draws(P)]
        row=dict(project=project,model=model,condition="T"+str(t),n=len(Y),delta_mcc=observed,**ci(samples))
        if project!="pooled":projectrq2.append(row);continue
        flips=rng.binomial(ct,0.5,size=(9999,len(ct)))
        P1=Y.sum();N=len(Y)-P1
        dt=flips@(diff*yy[:,None]);df=flips@(diff*(1-yy)[:,None])
        ma=metrics(A@Y+dt,A@(1-Y)+df,P1,N)[0];mb=metrics(B@Y-dt,B@(1-Y)-df,P1,N)[0]
        pv=(1+np.count_nonzero(np.abs((ma-mb).mean(axis=1))>=abs(observed)-1e-12))/10000
        ref=next(v for v in R["rq2"] if v["model"]==model and v["condition"]==row["condition"])
        assert ref["n"]==len(Y) and abs(ref["delta_mcc"]-observed)<1e-10
        rq2.append(dict(**row,p=pv,p_holm=None))
holm(rq2);out("rq2-fixed-project-bootstrap-and-permutation.csv",rq2);out("rq2-by-project-intervals.csv",projectrq2)
rq3=[];pairs=[];effects=[]
for t in range(1,4):
    common=np.all(d[:,:,t,:]>=0,axis=(0,1));blocks=[]
    ref=next(v for v in R["rq3"] if v["condition"]=="T"+str(t))
    assert ref["matched_commits"]==int(common.sum())
    for project in np.unique(proj[common]):
        sel=common&(proj==project);Y=y[sel]
        if len(set(Y))<2:continue
        W=np.ones(len(Y),int); vals=[]
        for i,model in enumerate(models):
            v=(confusion_scores(Y,d[i,:,0,:][:,sel],W)-confusion_scores(Y,d[i,:,t,:][:,sel],W)).mean()
            assert abs(ref["project_effects"][project][model]-v)<1e-10
            effects.append(dict(condition="T"+str(t),project=project,model=model,matched=len(Y),degradation=v));vals.append(v)
        blocks.append(vals)
    X=np.array(blocks);assert len(X)==ref["evaluable_projects"]
    ranks=np.rint(2*rankdata(X,axis=1)).astype(int)-4
    states={(0,0):1}
    for row in ranks:
        orders=Counter(tuple(x[:2]) for x in permutations(row.tolist()));new=Counter()
        for (a,b),n in states.items():
            for (c,e),mult in orders.items():new[(a+c,b+e)]+=n*mult
        states=new
    crit=sum(int(v)**2 for v in ranks.sum(0));tail=sum(n for (a,b),n in states.items() if a*a+b*b+(a+b)**2>=crit)
    denom=sum(states.values());assert denom==6**len(X)
    pval=tail/denom
    rq3.append(dict(condition="T"+str(t),matched_commits=int(common.sum()),project_blocks=len(X),tied_blocks=sum(len(set(x))<3 for x in X),friedman_stat=friedmanchisquare(*X.T).statistic,asymptotic_p=ref["friedman_p"],exact_randomization_p=pval,permutations=denom))
    for i,j in combinations(range(3),2):
        diff=X[:,i]-X[:,j];v=diff[diff!=0];pv=None
        if len(v)>=6 and pval<.05:
            r=np.rint(2*rankdata(abs(v))).astype(int); observed=abs(np.sign(v)@r)
            sums=np.array([0])
            for value in r:sums=np.r_[sums+value,sums-value]
            pv=float((abs(sums)>=observed).mean())
        pairs.append(dict(condition="T"+str(t),first=models[i],second=models[j],projects=len(X),nonzero=len(v),zeros=len(X)-len(v),abs_rank_ties=len(v)-len(set(abs(v))),mean_difference=diff.mean(),p=pv,p_holm=None))
holm(pairs);out("rq3-exact-friedman.csv",rq3);out("rq3-exact-wilcoxon.csv",pairs);out("rq3-matched-project-effects.csv",effects)
print("Supplement complete",rq3)
