"""Fresh expansion/confirmation after the preserved v3 family-transfer failure."""
def roster():
    out=[]
    def add(split,families,nus,grids):
        for family in families:
            for nu in nus:
                for nx,ny in grids:
                    out.append(dict(family=family,nu=nu,nx=nx,ny=ny,order=1,split=split,
                        name=f'{split}_{family}_n{nx}x{ny}_nu{nu:g}'.replace('.','p')))
    grids=[(8,8),(12,12),(16,8),(24,6)]
    add('train',['poly'],[.009,.018,.045],grids)
    add('train',['gaussian'],[.025,.055,.11],grids)
    add('validation',['mms','kov'],[.006,.022],[(10,10),(20,5)])
    add('validation',['poly'],[.012,.035],[(10,10),(20,5)])
    add('validation',['gaussian'],[.035,.075],[(10,10),(20,5)])
    add('family_test',['linke'],[.004,.012,.04,.12],[(10,10),(20,5)])
    add('geometry_test',['mms','kov','poly'],[.01,.025],[(18,10),(10,18)])
    add('geometry_test',['gaussian'],[.04,.08],[(18,10),(10,18)])
    for k in [2,3]:
        before=len(out);add('pressure_test',['mms'],[.006,.022],[(10,10),(20,5)])
        for c in out[before:]:c.update(k=k,name=c['name']+f'_k{k}')
    assert len(out)==72 and len({x['name']for x in out})==72
    return out

FEATURES=['umag','Pe','Pe_long','Pe_short','Pe_aniso','Snorm','Wnorm','Q','divu','gradnorm',
          'vsr','h','aspect','align','xi_coth','xi_long','xi_short',
          'scaled_pressure_gradient','pressure_stream_alignment','signed_divergence','boundary_distance']

def features(mesh,nu):
    """Computed predictor only; no exact field, target error, family or split."""
    import numpy as np
    import backend as B
    ids,x=B.V.R.features(mesh,nu);state,g=B.V.R.value_gradient(mesh,.5,.5)
    h=np.sqrt(np.prod(mesh['size'],axis=1));speed=np.linalg.norm(state[:,:2],axis=1)
    pg=g[:,2,:];pmag=np.linalg.norm(pg,axis=1)
    scaled=h*pmag/(speed*speed+nu*speed/h+1e-10)
    align=np.sum(pg*state[:,:2],axis=1)/np.maximum(pmag*speed,1e-10)
    div=g[:,0,0]+g[:,1,1]
    center=mesh['lo']+mesh['size']/2;lo=mesh['lo'].min(0);hi=(mesh['lo']+mesh['size']).max(0)
    distance=np.min(np.column_stack([center-lo,hi-center]),axis=1)/h
    out=np.column_stack([x,scaled,align,div,distance]);assert out.shape==(len(ids),21)and np.isfinite(out).all()
    return ids,out
