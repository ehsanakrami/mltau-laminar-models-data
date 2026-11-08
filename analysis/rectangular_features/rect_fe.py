"""Native rectangular Q1/Q2 interpolation, never a Q1 approximation of Q2.

Features use computed finite-element values/gradients only. Exact/reference
functions enter only the separate error integral supplied by its caller.
"""
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'qualification'))
from strict_features import read_state,corners_features

def basis(p,z):
    z=np.asarray(z)
    if p==1:return np.stack([1-z,z],-1),np.stack([-np.ones_like(z),np.ones_like(z)],-1)
    if p==2:return np.stack([2*z*z-3*z+1,4*z*(1-z),2*z*z-z],-1),np.stack([4*z-3,4-8*z,4*z-1],-1)
    raise ValueError('Only Q1/Q2 are supported')

def load(path):
    xy,conn,ids,t,fields=read_state(path)
    p={4:1,9:2}.get(conn.shape[1])
    if p is None or len(set(ids))!=len(ids):raise ValueError('Unsupported or duplicate elements')
    values=np.column_stack([fields[k][-1] for k in ['u','v','p']]);points=xy[conn]
    lo=points.min(1);hi=points.max(1);size=hi-lo
    if np.any(size<=0):raise ValueError('Degenerate element')
    q=np.linspace(0,1,p+1);xx,yy=np.meshgrid(q,q)
    expected=lo[:,None,:]+np.column_stack([xx.ravel(),yy.ravel()])[None,:,:]*size[:,None,:]
    distances=np.linalg.norm(expected[:,:,None,:]-points[:,None,:,:],axis=-1);order=distances.argmin(-1)
    if np.any(np.sort(order,axis=1)!=np.arange((p+1)**2)) or np.take_along_axis(distances,order[:,:,None],axis=2).max()>1e-10:raise ValueError('Nonrectangular/misordered element')
    nodevalues=values[conn[np.arange(len(ids))[:,None],order]].reshape(len(ids),p+1,p+1,3)
    if not np.isfinite(nodevalues).all():raise ValueError('Nonfinite nodal values')
    return dict(ids=ids,lo=lo,size=size,p=p,values=nodevalues,time=float(t[-1]))

def value_gradient(mesh,a,b):
    px,dx=basis(mesh['p'],a);py,dy=basis(mesh['p'],b);v=mesh['values']
    field=np.einsum('i,j,njik->nk',px,py,v)
    gx=np.einsum('i,j,njik->nk',dx,py,v)/mesh['size'][:,0,None]
    gy=np.einsum('i,j,njik->nk',px,dy,v)/mesh['size'][:,1,None]
    return field,np.stack([gx,gy],-1)

def sample(mesh,x,y):
    """Evaluate the actual FE polynomial at arbitrary points on a tensor mesh."""
    x,y=np.broadcast_arrays(x,y);shape=x.shape;x=x.ravel();y=y.ravel()
    lo=mesh['lo'];size=mesh['size'];xs=np.unique(lo[:,0]);ys=np.unique(lo[:,1])
    if len(xs)*len(ys)!=len(lo):raise ValueError('Expected complete tensor mesh')
    table=np.full((len(ys),len(xs)),-1,dtype=int)
    table[np.searchsorted(ys,lo[:,1]),np.searchsorted(xs,lo[:,0])]=np.arange(len(lo))
    if np.any(table<0):raise ValueError('Incomplete or overlapping tensor mesh')
    ix=np.clip(np.searchsorted(xs,x,side='right')-1,0,len(xs)-1)
    iy=np.clip(np.searchsorted(ys,y,side='right')-1,0,len(ys)-1);idx=table[iy,ix]
    a=(x-lo[idx,0])/size[idx,0];b=(y-lo[idx,1])/size[idx,1]
    if min(a.min(),b.min()) < -1e-10 or max(a.max(),b.max())>1+1e-10:raise ValueError('Point outside mesh')
    px,_=basis(mesh['p'],a);py,_=basis(mesh['p'],b)
    return np.einsum('ni,nj,njik->nk',px,py,mesh['values'][idx]).reshape(*shape,3)

def features(mesh,nu):
    field,grad=value_gradient(mesh,.5,.5);rows=[]
    for u,g,(hx,hy) in zip(field[:,:2],grad[:,:2,:],mesh['size']):
        # These virtual corners encode the exact center value and gradient;
        # they do not replace a Q2 field by its four physical corner values.
        corners=np.stack([u-hx*g[:,0]/2-hy*g[:,1]/2,u+hx*g[:,0]/2-hy*g[:,1]/2,
                          u+hx*g[:,0]/2+hy*g[:,1]/2,u-hx*g[:,0]/2+hy*g[:,1]/2])
        rows.append(corners_features(corners,hx,hy,nu))
    return mesh['ids'],np.asarray(rows)

def integrate(mesh,exact,order=20):
    v=mesh['values'].copy();origin=v[...,2].mean();v[...,2]-=origin
    m=dict(mesh,values=v);area=np.prod(mesh['size'],axis=1);volume=area.sum()
    z,w=np.polynomial.legendre.leggauss(order);z=(z+1)/2;w=w/2
    errors=np.zeros(3);pmean=0.;div2=0.;pressure=[]
    for a,wa in zip(z,w):
        for b,wb in zip(z,w):
            f,g=value_gradient(m,a,b);xy=m['lo']+np.array([a,b])*m['size'];weights=wa*wb*area
            delta=f-exact(xy[:,0],xy[:,1]);errors+=(weights[:,None]*delta**2).sum(0);pmean+=np.dot(weights,delta[:,2])
            pressure.append((wa*wb,delta[:,2].copy()))
            div2+=np.dot(weights,(g[:,0,0]+g[:,1,1])**2)
    # Center before squaring: subtracting two O(1) second moments loses digits
    # when the actual mean-free pressure error is small (not quadrature error).
    centered=pmean/volume
    p2=sum(weight*np.dot(area,(delta-centered)**2) for weight,delta in pressure)
    return dict(velocity_L2=float(np.sqrt(errors[:2].sum())),p_L2=float(np.sqrt(p2)),
                div_L2=float(np.sqrt(div2)),p_gauge=float(pmean/volume+origin))
