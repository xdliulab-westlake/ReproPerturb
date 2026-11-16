"""Condition-balanced composition metrics; no model fitting or silent repair."""
import numpy as np


def validate(p):
    p=np.asarray(p,dtype=np.float64)
    if p.ndim!=2 or not np.isfinite(p).all() or (p<0).any() or (p>1).any():
        raise ValueError('Expected finite nonnegative composition rows')
    if not np.allclose(p.sum(1),1,atol=1e-5,rtol=0):
        raise ValueError('Composition rows must sum to one')
    return p


def align(ids,classes,pred,expected_ids,expected_classes):
    ids,classes=list(ids),list(classes)
    if len(set(ids))!=len(ids) or len(set(classes))!=len(classes):raise ValueError('Duplicate keys')
    if set(ids)!=set(expected_ids) or set(classes)!=set(expected_classes):raise ValueError('Missing/extra keys')
    pred=validate(pred)
    if pred.shape!=(len(ids),len(classes)):raise ValueError('Shape differs from keys')
    return pred[np.ix_([ids.index(x) for x in expected_ids],[classes.index(x) for x in expected_classes])]


def clr(p,epsilon=1e-4):
    a=np.log(np.asarray(p,dtype=np.float64)+epsilon)
    return a-a.mean(axis=-1,keepdims=True)


def softmax(z):
    v=np.exp(z-np.max(z,axis=-1,keepdims=True))
    return v/v.sum(axis=-1,keepdims=True)


def clr_additive(control,singletons,epsilon=1e-4):
    validate(np.asarray(control)[None]);validate(singletons)
    return softmax(clr(singletons,epsilon).sum(0)-(len(singletons)-1)*clr(control,epsilon))


def simplex_projection(v):
    v=np.asarray(v,dtype=np.float64)
    u=np.sort(v)[::-1];css=np.cumsum(u)-1
    j=np.arange(1,len(u)+1);rho=np.flatnonzero(u-css/j>0)[-1]
    return np.maximum(v-css[rho]/(rho+1),0)


def correlation(x,y,rank=False):
    if len(x)<3 or np.std(x)<=1e-8 or np.std(y)<=1e-8:return None
    if rank:
        def ranks(a):
            _,inverse,counts=np.unique(a,return_inverse=True,return_counts=True)
            average=np.cumsum(counts)-(counts-1)/2
            return average[inverse]
        x,y=ranks(x),ranks(y)
    return float(np.corrcoef(x,y)[0,1])


def rank_metrics(pred,truth,k=3):
    pred,truth=np.asarray(pred),np.asarray(truth)
    if len(truth)<k:raise ValueError('Not enough candidate conditions')
    if np.ptp(truth)<=1e-8:return dict(ndcg=None,yield_pp=None,regret_pp=None)
    order=np.argsort(-pred,kind='stable');scores=pred[order];y=truth[order]
    expected=np.empty(len(y),dtype=float)
    for score in np.unique(scores):
        mask=scores==score;expected[mask]=y[mask].mean()
    discount=1/np.log2(np.arange(2,k+2));ideal=np.sort(truth)[::-1][:k]
    idcg=float(ideal@discount);yield_=float(expected[:k].mean())
    return dict(ndcg=float(expected[:k]@discount/idcg) if idcg else None,
                yield_pp=100*yield_,regret_pp=max(0.,100*(float(ideal.mean())-yield_)))


def metrics(pred,truth):
    p,q=validate(pred),validate(truth)
    if p.shape!=q.shape:raise ValueError('Shape mismatch')
    mid=(p+q)/2
    def kl(a):
        mask=a>0;out=np.zeros_like(a);out[mask]=a[mask]*np.log2(a[mask]/mid[mask]);return out.sum(1)
    return dict(mae_pp=float(100*np.abs(p-q).mean()),rmse_pp=float(100*np.sqrt(((p-q)**2).mean())),
                jsd_bits=float((.5*kl(p)+.5*kl(q)).mean()),
                per_condition_mae_pp=(100*np.abs(p-q).mean(1)).tolist(),
                per_type_mae_pp=(100*np.abs(p-q).mean(0)).tolist(),
                per_type_bias_pp=(100*(p-q).mean(0)).tolist(),
                per_type_pearson=[correlation(p[:,j],q[:,j]) for j in range(p.shape[1])],
                per_type_spearman=[correlation(p[:,j],q[:,j],True) for j in range(p.shape[1])])


def response_metrics(pred,truth,control,tau=.02):
    d=validate(truth)-validate(control);e=validate(pred)-control
    active=np.abs(d)>=tau;action=np.abs(e)>=tau
    correct=active&action&(np.sign(d)==np.sign(e))
    out=[]
    for j in range(d.shape[1]):
        n=int(active[:,j].sum());quiet=~active[:,j]
        out.append(dict(n_active=n,direction_accuracy=float(correct[:,j].sum()/n) if n else None,
          abstain_rate=float((active[:,j]&~action[:,j]).sum()/n) if n else None,
          false_action_rate=float(action[quiet,j].mean()) if quiet.any() else None,
          delta_pearson=correlation(d[:,j],e[:,j]),delta_spearman=correlation(d[:,j],e[:,j],True)))
    return out
