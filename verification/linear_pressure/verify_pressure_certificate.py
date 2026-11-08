"""Independent support replay with outward scalar roundoff bounds.
Certificate concerns exact real values of pinned binary64 affine arrays only.
Uses no claimed accuracy of an SVD: arbitrary computed factors plus residual
provide an upper bound on each nuclear norm by a norm inequality.
"""
from pathlib import Path
from decimal import Decimal,localcontext,ROUND_CEILING,ROUND_FLOOR
import datetime,json,os,signal,time,traceback
import numpy as np
from scipy.linalg import svd
D=Decimal;UNIT=D.from_float(2.**-53)
TINY_ALLOWANCE=D('1e-200')

def dec(v):return D.from_float(float(v))
def upsum(*v):
 with localcontext() as c:c.prec=80;c.rounding=ROUND_CEILING;return sum(v,D(0)).next_plus()
def upprod(*v):
 with localcontext() as c:
  c.prec=80;c.rounding=ROUND_CEILING;r=D(1)
  for a in v:r*=a
  return r.next_plus()
def lowsub(a,b):
 with localcontext() as c:c.prec=80;c.rounding=ROUND_FLOOR;return (a-b).next_minus()
def sqroot(v):
 with localcontext() as c:c.prec=80;c.rounding=ROUND_CEILING;return v.sqrt().next_plus()
def gamma(k):
 with localcontext() as c:c.prec=80;c.rounding=ROUND_CEILING;u=D(k)*UNIT;assert u<D('.01');return (u/(1-u)).next_plus()
def quotient(a,b):
 with localcontext() as c:c.prec=80;c.rounding=ROUND_CEILING;return (a/b).next_plus()
def normF(a):
 a=np.asarray(a,dtype=np.float64);assert np.isfinite(a).all();v=float(np.sum(a*a,dtype=np.float64));assert np.isfinite(v)
 # 2N+2 allows all products and any conventional positive reduction order.
 assert a.size<10**9 and (not a.size or np.max(np.abs(a))<1e100)
 return upsum(sqroot(quotient(dec(v),lowsub(D(1),gamma(2*a.size+2)))),D('1e-150'))
def dotbound(a,b):
 p=a*b;value=float(np.sum(p,dtype=np.float64));magnitude=float(np.sum(abs(p),dtype=np.float64));k=gamma(2*a.size+2)
 err=upsum(upprod(k,quotient(dec(magnitude),lowsub(D(1),k))),TINY_ALLOWANCE)
 return dec(value),err

def nuclear_upper(a):
 # Valid for ANY factors X,Y; their construction's singular-value accuracy is irrelevant.
 u,s,vt=svd(a,full_matrices=False,lapack_driver='gesvd');root=np.sqrt(np.maximum(s,0.));X=u*root[None,:];Y=vt.T*root[None,:]
 prod=X@Y.T;res=a-prod;fx,fy=normF(X),normF(Y)
 product_error=upprod(gamma(X.shape[1]),fx,fy)
 subtract_error=upprod(gamma(1),upsum(normF(a),normF(prod)))
 residual_upper=upsum(normF(res),product_error,subtract_error)
 bound=upsum(upprod(fx,fy),upprod(sqroot(D(min(a.shape))),residual_upper))
 return bound,dict(factor_product=float(upprod(fx,fy)),residual_upper=float(residual_upper))

def dual_upper(y,key):
 if key in [9,10,11,'named']:return upsum(*(normF(y[:,j]) for j in range(y.shape[1])))
 if key==12:return normF(y)
 return nuclear_upper(y)[0]


def main():
 root=Path(__file__).resolve().parent
 with np.load(root/'pressure_certificate_data.npz',allow_pickle=False) as z:E,A,B,Y=[z[k] for k in ['E','A','B','Y']]
 assert E.shape==Y.shape==(464,57) and A.shape==(464,464) and B.shape==(57,57)
 t=A.T@Y;g=t@B.T
 error=upsum(upprod(gamma(Y.shape[1]),normF(t),normF(B)),upprod(gamma(A.shape[0]),normF(A),normF(Y),normF(B)),D('1e-200'))
 value,doterror=dotbound(Y,E);yn,_=nuclear_upper(Y);gn,_=nuclear_upper(g)
 adjoint_upper=upsum(gn,upprod(sqroot(D(57)),error));constant_lower=lowsub(lowsub(value,doterror),yn);gap=lowsub(constant_lower,adjoint_upper)
 result=dict(positive_separation=gap>0,constant_lower=str(constant_lower),adjoint_nuclear_upper=str(adjoint_upper),separation_lower=str(gap),scope='Pinned finite binary64 linear-response model only; no nonlinear-closure or continuum infeasibility claim; no core solver, assembly or training reproduction.')
 print(json.dumps(result,indent=2));return 0 if gap>0 else 2
if __name__=='__main__':raise SystemExit(main())
