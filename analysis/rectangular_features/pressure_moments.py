import numpy as np
from roster import features,FEATURES
NAMES=FEATURES+['pressure_cell_rms','pressure_mixed_curvature','velocity_mixed_curvature']
def augment(mesh,nu):
    import backend as B
    assert mesh['p']==1
    ids,base=features(mesh,nu);state,_=B.V.R.value_gradient(mesh,.5,.5)
    v=mesh['values'];h=np.sqrt(np.prod(mesh['size'],axis=1));speed=np.linalg.norm(state[:,:2],axis=1)
    mixed=(v[:,1,1]-v[:,1,0]-v[:,0,1]+v[:,0,0])/np.prod(mesh['size'],axis=1)[:,None]
    p=v[...,2];prms=np.sqrt(np.mean((p-p.mean((1,2))[:,None,None])**2,axis=(1,2)))
    dp=speed*speed+nu*speed/h+(nu/h)**2+1e-12;du=speed+nu/h+1e-12
    extra=np.column_stack([prms/dp,h*h*np.abs(mixed[:,2])/dp,h*h*np.linalg.norm(mixed[:,:2],axis=1)/du])
    out=np.column_stack([base,extra]);assert np.isfinite(out).all()
    return ids,out
