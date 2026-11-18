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


def correlation(x,y,rank=False):
    if len(x)<3 or np.std(x)<=1e-8 or np.std(y)<=1e-8:return None
    if rank:
        def ranks(a):
            _,inverse,counts=np.unique(a,return_inverse=True,return_counts=True)
            average=np.cumsum(counts)-(counts-1)/2
            return average[inverse]
        x,y=ranks(x),ranks(y)
    return float(np.corrcoef(x,y)[0,1])


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


def score(prediction, truth):
    """Evaluate one seed across aligned test conditions."""
    p, q = np.asarray(prediction, dtype=float), validate(truth)
    if p.shape != q.shape or np.any(p < 0) or not np.allclose(p.sum(1), 1, atol=1e-4, rtol=0):
        raise ValueError("Expected aligned composition rows summing to one")
    p = p / p.sum(1, keepdims=True)
    result = metrics(p, q)
    result["hellinger_distance"] = float(np.mean(np.linalg.norm(np.sqrt(p) - np.sqrt(q), axis=1) / np.sqrt(2)))
    predicted = np.isclose(p, p.max(1, keepdims=True), atol=1e-12, rtol=0)
    observed = np.isclose(q, q.max(1, keepdims=True), atol=1e-12, rtol=0)
    result["dominant_accuracy_pct"] = float(100 * np.mean((predicted & observed).sum(1) / predicted.sum(1)))
    return result
