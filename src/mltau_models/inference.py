"""NumPy-only inference for the source-free model/data candidate.

The caller must identify the feature contract, not merely its array width.
This is model inference, not a substitute for the private fluid solver.
"""
import numpy as np
LOW,HIGH=np.log(.004),np.log(30.)
def predict(model,features,feature_variant):
    if feature_variant!=model['feature_variant']:raise ValueError('Feature contract mismatch')
    if np.ma.isMaskedArray(features)and np.any(np.ma.getmaskarray(features)):raise ValueError('Masked features')
    x=np.asarray(features,dtype=float);mean=np.asarray(model['transform']['mean']);scale=np.asarray(model['transform']['scale'])
    if x.ndim!=2 or x.shape[1]!=len(mean)or mean.shape!=scale.shape:raise ValueError('Feature dimension mismatch')
    if not np.isfinite(x).all()or not np.isfinite(mean).all()or not np.isfinite(scale).all()or np.any(scale<=0):raise ValueError('Invalid normalization/input')
    x=np.clip((np.arcsinh(x)-mean)/scale,-8,8);p=[np.asarray(v,dtype=float)for v in model['parameters']]
    if not all(np.isfinite(v).all()for v in p):raise ValueError('Invalid parameters')
    if model['kind']=='constant':z=np.repeat(p[0].item(),len(x))
    elif model['kind']=='affine':z=np.einsum('ij,j->i',x,p[0])+p[1]
    elif model['kind']=='tanh':
        hidden=np.tanh(np.einsum('ij,jk->ik',x,p[0])+p[1]);z=np.einsum('ij,j->i',hidden,p[2])+p[3]
    else:raise ValueError('Unsupported architecture')
    return np.exp(np.clip(z,LOW,HIGH))

def constant_coefficient(model):
    if model['kind']!='constant':raise ValueError('Not a constant model')
    return float(np.exp(np.clip(np.asarray(model['parameters'][0]).item(),LOW,HIGH)))
