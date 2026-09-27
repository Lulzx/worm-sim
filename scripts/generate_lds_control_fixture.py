#!/usr/bin/env python3
"""Independent dense joint-Gaussian oracle for controlled LDS smoothing and one EM step."""
import json
from pathlib import Path
import numpy as np

A=np.array([[0.8,0.15],[-0.1,0.7]])
B=np.array([[0.3,-0.4],[0.2,0.1]])
C=np.array([[1.0,0.3],[-0.2,0.8]])
Q=np.array([[0.12,0.02],[0.02,0.09]])
P0=np.array([[0.7,0.1],[0.1,0.5]])
R=np.array([0.15,0.25])
u=np.array([[0.2,1.0],[-0.7,0.0],[0.8,1.0],[9.0,-9.0]])
sequence=[[(0,0.4,1.0),(1,-0.3,0.5)],[(0,-0.2,0.8)],[(1,0.9,1.0)],[(0,0.1,0.6),(1,0.5,0.9)]]
T,k,d=len(sequence),2,2
mean=np.zeros((T,k));prior=np.zeros((T*k,T*k));prior[:k,:k]=P0
for t in range(1,T):
    mean[t]=A@mean[t-1]+B@u[t-1]
    prior[t*k:(t+1)*k,t*k:(t+1)*k]=A@prior[(t-1)*k:t*k,(t-1)*k:t*k]@A.T+Q
    for s in range(t):
        prior[t*k:(t+1)*k,s*k:(s+1)*k]=A@prior[(t-1)*k:t*k,s*k:(s+1)*k]
        prior[s*k:(s+1)*k,t*k:(t+1)*k]=prior[t*k:(t+1)*k,s*k:(s+1)*k].T
rows=[];values=[];noise=[]
for t,obs in enumerate(sequence):
    for i,y,w in obs:
        row=np.zeros(T*k);row[t*k:(t+1)*k]=C[i];rows.append(row);values.append(y);noise.append(R[i]/w)
H=np.array(rows);y=np.array(values);S=H@prior@H.T+np.diag(noise);residual=y-H@mean.ravel()
post_mean=(mean.ravel()+prior@H.T@np.linalg.solve(S,residual)).reshape(T,k)
post_cov=prior-prior@H.T@np.linalg.solve(S,H@prior)
nll=0.5*(len(y)*np.log(2*np.pi)+np.linalg.slogdet(S)[1]+residual@np.linalg.solve(S,residual))
cov=[post_cov[t*k:(t+1)*k,t*k:(t+1)*k] for t in range(T)]
lag=[post_cov[(t+1)*k:(t+2)*k,t*k:(t+1)*k] for t in range(T-1)]
# Independent expected squared residual calculations using the joint posterior.
ridge=0.002;cap=0.4;gram=np.zeros((k+d,k+d));cross=np.zeros((k,k+d))
for t in range(T-1):
    z=np.r_[post_mean[t],u[t]]
    gram+=np.outer(z,z);gram[:k,:k]+=cov[t]
    cross+=np.outer(post_mean[t+1],z);cross[:,:k]+=lag[t]
penalty=ridge*(T-1)
AB=np.linalg.solve(gram+penalty*np.eye(k+d),cross.T).T
unprojected_norm=float(np.linalg.svd(AB[:,:k],compute_uv=False)[0])
newA=AB[:,:k]*min(1.0,cap/unprojected_norm)
newB=np.linalg.solve(gram[k:,k:]+penalty*np.eye(d),(cross[:,k:]-newA@gram[:k,k:]).T).T
newQ=np.zeros((k,k))
for t in range(T-1):
    residual=post_mean[t+1]-newA@post_mean[t]-newB@u[t]
    newQ+=np.outer(residual,residual)+cov[t+1]+newA@cov[t]@newA.T-lag[t]@newA.T-newA@lag[t].T
newQ/=T-1;newQ+=1e-6*np.eye(k)
newC=np.zeros_like(C);newR=np.zeros_like(R)
for i in range(len(R)):
    obs=[(t,y,w) for t,frame in enumerate(sequence) for j,y,w in frame if i==j]
    gramC=sum((w*(cov[t]+np.outer(post_mean[t],post_mean[t])) for t,y,w in obs),start=np.zeros((k,k)))
    crossC=sum((w*y*post_mean[t] for t,y,w in obs),start=np.zeros(k))
    newC[i]=np.linalg.solve(gramC+ridge*len(obs)*np.eye(k),crossC)
    newR[i]=max(1e-4,sum(w*((y-newC[i]@post_mean[t])**2+newC[i]@cov[t]@newC[i]) for t,y,w in obs)/len(obs))
newP0=cov[0]+np.outer(post_mean[0],post_mean[0])+1e-6*np.eye(k)
def array(x):return x.ravel().tolist()
def model(a,b,c,q,r,p):return {'dim':k,'outputs':len(R),'input_dim':d,'transition':array(a),'input_weights':array(b),'observation':array(c),'process_cov':array(q),'noise':array(r),'initial_cov':array(p)}
fixture={'description':'Independent NumPy full joint Gaussian conditioning, no Kalman recursion; direct expected residual moment M-step with active A projection and conditional B refit. Last input row is deliberately unused.', 'numpy_version':np.__version__,'gaussian':model(A,B,C,Q,R,P0),'sequence':sequence,'inputs':u.tolist(),'means':post_mean.tolist(),'covariances':[array(x) for x in cov],'lag_covariances':[array(x) for x in lag],'negative_log_likelihood':float(nll),'ridge':ridge,'cap':cap,'unprojected_transition_norm':unprojected_norm,'em':model(newA,newB,newC,newQ,newR,newP0)}
Path('tests/fixtures/lds_controlled.json').write_text(json.dumps(fixture,indent=2)+'\n')
print('unprojected norm',unprojected_norm,'cap',cap)
