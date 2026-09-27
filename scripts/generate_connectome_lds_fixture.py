#!/usr/bin/env python3
"""Independent joint-Gaussian + full parameter-normal-system oracle for tied-input EM."""
import json
from pathlib import Path
import numpy as np

def mm(a, b):
    a, b = np.asarray(a), np.asarray(b)
    subscripts = {(1,1): 'i,i->', (2,1): 'ij,j->i', (1,2): 'i,ij->j', (2,2): 'ij,jk->ik'}[(a.ndim,b.ndim)]
    return np.einsum(subscripts, a, b, optimize=False)

n = 3
lags = 3
A = np.array([[0.8, 0.0, -0.1], [0.2, 0.75, 0.0], [0.0, -0.15, 0.7]])
Q = np.diag([0.12, 0.09, 0.15])
R = np.array([0.1, 0.2, 0.15])
P0 = np.diag([0.3, 0.2, 0.4])
kernel = np.array([0.3, 0.15, -0.1])
allowed = [[0, 2], [0, 1], [1, 2]]
sequences = [{'target': 0, 'observations': [[(0, 0.1, 1.0), (2, -0.2, 0.7)], [(0, 0.4, 1.0), (1, 0.2, 0.8)], [(1, 0.5, 1.0), (2, 0.1, 1.0)], [(0, 0.2, 0.6), (2, 0.3, 1.0)]]}, {'target': 1, 'observations': [[(0, -0.1, 1.0), (1, 0.2, 0.5)], [(1, 0.7, 1.0)], [(0, 0.2, 0.8), (2, -0.3, 1.0)], [(1, 0.4, 1.0), (2, -0.1, 0.7)]]}]
posteriors = []
nll = 0.0
observations = 0
transitions = sum((len(s['observations']) - 1 for s in sequences))
for seq in sequences:
    T = len(seq['observations'])
    mean = np.zeros((T, n))
    prior = np.zeros((T * n, T * n))
    prior[:n, :n] = P0
    for t in range(1, T):
        mean[t] = mm(A, mean[t - 1])
        mean[t, seq['target']] += kernel[t - 1]
        prior[t * n:(t + 1) * n, t * n:(t + 1) * n] = mm(mm(A, prior[(t - 1) * n:t * n, (t - 1) * n:t * n]), A.T) + Q
        for s in range(t):
            prior[t * n:(t + 1) * n, s * n:(s + 1) * n] = mm(A, prior[(t - 1) * n:t * n, s * n:(s + 1) * n])
            prior[s * n:(s + 1) * n, t * n:(t + 1) * n] = prior[t * n:(t + 1) * n, s * n:(s + 1) * n].T
    rows = []
    values = []
    noise = []
    for t, frame in enumerate(seq['observations']):
        for i, y, w in frame:
            row = np.zeros(T * n)
            row[t * n + i] = 1.0
            rows.append(row)
            values.append(y)
            noise.append(R[i] / w)
    H = np.array(rows)
    values = np.array(values)
    S = mm(mm(H, prior), H.T) + np.diag(noise)
    res = values - mm(H, mean.ravel())
    m = (mean.ravel() + mm(mm(prior, H.T), np.linalg.solve(S, res))).reshape(T, n)
    cov = prior - mm(mm(prior, H.T), np.linalg.solve(S, mm(H, prior)))
    nll += 0.5 * (len(values) * np.log(2 * np.pi) + np.linalg.slogdet(S)[1] + mm(res, np.linalg.solve(S, res)))
    observations += len(values)
    posteriors.append((m, cov))
indices = [(i, j) for i, row in enumerate(allowed) for j in row]
d = len(indices) + lags
ridge = 0.002
cap = 0.4
gram = ridge * transitions * np.eye(d)
rhs = np.zeros(d)
for seq, (m, p) in zip(sequences, posteriors):
    for t in range(len(m) - 1):
        V = p[t * n:(t + 1) * n, t * n:(t + 1) * n]
        lag = p[(t + 1) * n:(t + 2) * n, t * n:(t + 1) * n]
        for i in range(n):
            M = np.zeros((d, n))
            constant = np.zeros(d)
            for k, (row, col) in enumerate(indices):
                if row == i:
                    M[k, col] = 1
            if i == seq['target']:
                constant[len(indices) + t] = 1
            f = mm(M, m[t]) + constant
            gram += (mm(mm(M, V), M.T) + np.outer(f, f)) / Q[i, i]
            rhs += (mm(M, lag[i]) + f * m[t + 1, i]) / Q[i, i]
coeff = np.linalg.solve(gram, rhs)
newA = np.zeros_like(A)
for k, (i, j) in enumerate(indices):
    newA[i, j] = coeff[k]
rawA = newA.copy()
rawkernel = coeff[len(indices):].copy()
unprojected = float(np.linalg.svd(newA, compute_uv=False)[0])
newA *= min(1.0, cap / unprojected)
newkernel = coeff[len(indices):]
if unprojected > cap:
    G = ridge * transitions * np.eye(lags)
    b = np.zeros(lags)
    for seq, (m, p) in zip(sequences, posteriors):
        i = seq['target']
        for t in range(len(m) - 1):
            feature = np.eye(lags)[t]
            G += np.outer(feature, feature) / Q[i, i]
            b += feature * (m[t + 1, i] - mm(newA[i], m[t])) / Q[i, i]
    newkernel = np.linalg.solve(G, b)
newQ = np.zeros(n)
newP = np.zeros(n)
newR = np.zeros(n)
count = np.zeros(n)
for seq, (m, p) in zip(sequences, posteriors):
    newP += np.diag(p[:n, :n]) + m[0] ** 2
    for t, frame in enumerate(seq['observations']):
        for i, y, w in frame:
            newR[i] += w * ((y - m[t, i]) ** 2 + p[t * n + i, t * n + i])
            count[i] += 1
    for t in range(len(m) - 1):
        V = p[t * n:(t + 1) * n, t * n:(t + 1) * n]
        V1 = p[(t + 1) * n:(t + 2) * n, (t + 1) * n:(t + 2) * n]
        lag = p[(t + 1) * n:(t + 2) * n, t * n:(t + 1) * n]
        r = m[t + 1] - mm(newA, m[t])
        r[seq['target']] -= newkernel[t]
        newQ += r * r + np.diag(V1 + mm(mm(newA, V), newA.T) - mm(lag, newA.T) - mm(newA, lag.T))
newQ = np.maximum(newQ / transitions, 1e-06)
newP = np.maximum(newP / len(sequences), 1e-06)
newR = np.maximum(newR / count, 1e-06)

def gaussian(a, q, r, p):
    return {'dim': n, 'outputs': n, 'transition': a.ravel().tolist(), 'input_dim': n, 'input_weights': np.eye(n).ravel().tolist(), 'observation': np.eye(n).ravel().tolist(), 'process_cov': np.asarray(q).ravel().tolist(), 'noise': r.tolist(), 'initial_cov': np.asarray(p).ravel().tolist()}
fixture = {'description': 'Full joint Gaussian conditioning; dense normal solve with shared lag coefficients; active norm cap and conditional shared-kernel refit. Target 2 is never stimulated in training.', 'numpy_version': np.__version__, 'model': {'gaussian': gaussian(A, Q, R, P0), 'allowed': allowed, 'kernel': kernel.tolist(), 'sample_dt': 0.5}, 'sequences': sequences, 'ridge': ridge, 'cap': cap, 'unprojected_transition': rawA.ravel().tolist(), 'unprojected_kernel': rawkernel.tolist(), 'unprojected_transition_norm': unprojected, 'preceding_negative_log_likelihood': float(nll), 'observations': observations, 'expected': {'gaussian': gaussian(newA, np.diag(newQ), newR, np.diag(newP)), 'allowed': allowed, 'kernel': newkernel.tolist(), 'sample_dt': 0.5}}
Path('tests/fixtures/connectome_lds.json').write_text(json.dumps(fixture, indent=2, allow_nan=False) + '\n')
print('unprojected norm', unprojected, 'kernel', newkernel)
