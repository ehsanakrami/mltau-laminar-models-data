"""Continuous directional-metric replacement of six axis-ranked descriptors.

No eigenvector or long-axis choice is made, including at an isotropic metric.
The guaranteed coordinate symmetry is axis swaps/reflections on Q1 rectangles.
"""
import numpy as np
from pressure_moments import augment as legacy,NAMES as OLD_NAMES
NAMES=list(OLD_NAMES)
for i,n in {2:'Pe_stream',3:'Pe_cross',4:'Pe_direction_ratio',13:'metric_velocity_contrast',15:'xi_stream',16:'xi_cross'}.items():NAMES[i]=n

def xi(p):
    """Finite, continuous coth(p)-1/p extension with a smooth series blend.

    Avoid subtracting two ~1/p numbers near zero and avoid a hard unmatched
    series switch. The series error at .02 is below float64 roundoff.
    """
    p=np.asarray(p,dtype=float)
    if np.any(p<0)or not np.isfinite(p).all():raise ValueError('Invalid Peclet number')
    z=np.minimum(p,.02)
    small=z/3-z**3/45+2*z**5/945-z**7/4725+2*z**9/93555
    q=np.maximum(p,.01)
    large=1+2*np.exp(-2*q)/(-np.expm1(-2*q))-1/q
    t=np.clip((p-.01)/.01,0,1);w=t*t*(3-2*t)
    return np.where(p<=.01,small,np.where(p>=.02,large,(1-w)*small+w*large))

def augment(mesh,nu):
    import backend as B
    for key in ['lo','size','values']:
        a=mesh[key]
        if np.ma.isMaskedArray(a)and np.any(np.ma.getmaskarray(a)):raise ValueError('Masked geometry or field')
    ids,raw=legacy(mesh,nu);out=np.asarray(raw,dtype=float).copy()
    state,_=B.V.R.value_gradient(mesh,.5,.5);u=np.asarray(state[:,:2]);hx,hy=np.asarray(mesh['size']).T
    speed2=np.sum(u*u,axis=1);stream=np.hypot(u[:,0]/hx,u[:,1]/hy);cross=np.hypot(u[:,1]/hx,u[:,0]/hy)
    pl=np.divide(speed2,2*nu*stream,out=np.zeros_like(speed2),where=stream>0)
    ps=np.divide(speed2,2*nu*cross,out=np.zeros_like(speed2),where=cross>0)
    h=np.sqrt(hx*hy);velocity_floor2=(1e-12*nu/h)**2
    contrast=(hx*hx-hy*hy)*(u[:,0]**2-u[:,1]**2)/((hx*hx+hy*hy)*(speed2+velocity_floor2))
    out[:,2]=pl;out[:,3]=ps
    out[:,4]=np.minimum(np.maximum(pl,ps)/np.maximum(np.minimum(pl,ps),1e-6),100)
    out[:,13]=contrast;out[:,14]=xi(np.sqrt(speed2)*h/(2*nu));out[:,15]=xi(pl);out[:,16]=xi(ps)
    assert np.isfinite(out).all();return ids,out
