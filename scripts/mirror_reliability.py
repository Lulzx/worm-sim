#!/usr/bin/env python3
"""Left/right test-retest reliability of c302 synapse counts.

For a neuron with a mirrored partner (AVAL/AVAR), the connection a->b and its
mirror m(a)->m(b) are two observations of the same wiring. Left/right disagreement
mixes reconstruction error, genuine asymmetry (ASE, AWC) and, where the source
reconstruction pooled animals, inter-animal variability. Every estimate here is
therefore a lower bound on measurement precision, not a pure error estimate.

Presence is compared with degree-preserving rewired graphs, in which each edge
keeps its count. Counts follow a censored Poisson-lognormal model: the two sides
share a latent log-rate theta ~ N(mu, tau^2), each side adds N(0, sigma^2), each
side independently drops out with a probability that depends on its rate, surviving counts are Poisson,
and a pair enters the sample only when one side has a contact. The
fit gives an empirical-Bayes count per edge that pools its mirror where defined.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln, logsumexp

BINS = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 9), (10, 19), (20, 10**9)]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mirror_map(names):
    """Swap a terminal L/R suffix only when the swapped cell exists."""
    out = {}
    for x in names:
        y = x[:-1]+{'L': 'R', 'R': 'L'}.get(x[-1], x[-1])
        out[x] = y if y != x and y in names else x
    return out


def key(a, b, directed):
    return (a, b) if directed or a <= b else (b, a)


def mirror_pairs(counts, mirror, directed):
    """Unordered mirror pairs with at least one observed side; self-mirrors excluded."""
    pairs = {}
    for a, b in counts:
        own, other = key(a, b, directed), key(mirror[a], mirror[b], directed)
        if own != other:
            pairs[tuple(sorted((own, other)))] = None
    return [(x, y, counts.get(x, 0.), counts.get(y, 0.)) for x, y in sorted(pairs)]


def rewire(edges, directed, rng, sweeps=10):
    """Degree-preserving double-edge swaps; counts travel with their edges."""
    edges = [tuple(e) for e in edges]
    present = set(edges)
    for _ in range(sweeps*len(edges)):
        i, j = rng.integers(len(edges), size=2)
        (a, b), (c, d) = edges[i], edges[j]
        if not directed and rng.random() < .5:
            c, d = d, c
        x, y = key(a, d, directed), key(c, b, directed)
        if i == j or x == y or x in present or y in present or x[0] == x[1] or y[0] == y[1]:
            continue
        present.difference_update((edges[i], edges[j]))
        present.update((x, y))
        edges[i], edges[j] = x, y
    return edges


def bin_of(count):
    # Half-open bins, so averaged half-integer gap sizes fall in the lower bin.
    return next(k for k, (lo, hi) in enumerate(BINS) if lo <= count < hi+1)


def presence_by_bin(counts, mirror, directed):
    hit, total = np.zeros(len(BINS)), np.zeros(len(BINS))
    for (a, b), c in counts.items():
        other = key(mirror[a], mirror[b], directed)
        if other != (a, b):
            k = bin_of(c)
            total[k] += 1
            hit[k] += other in counts
    return hit, total


class Model:
    """Censored zero-inflated bivariate Poisson-lognormal over mirror pairs.

    Each side independently drops out with probability q(eta) = sigmoid(c0 + c1*eta)
    (a structural zero: asymmetry or an omitted contact), otherwise its count is
    Poisson around its rate. A constant q underpredicts mirror presence for strong
    edges and overpredicts it for weak ones. Integrals use Gauss-Hermite quadrature.
    """

    def __init__(self, y1, y2, nodes=40):
        # Count pairs repeat heavily; the likelihood is exact over unique pairs.
        unique, self.multiplicity = np.unique(np.c_[y1, y2], axis=0, return_counts=True)
        self.y1, self.y2 = unique[:, 0], unique[:, 1]
        z, w = np.polynomial.hermite_e.hermegauss(nodes)
        self.z1, self.z2 = np.meshgrid(z, z, indexing='ij')
        self.logw = np.log(np.outer(w, w)/w.sum()**2)

    @staticmethod
    def unpack(params):
        return params[0], np.exp(params[1]), np.exp(params[2]), (params[3], params[4])

    def grid(self, params):
        mu, tau, sigma, _ = self.unpack(params)
        # Shared theta plus independent side noise, by explicit construction.
        return mu+tau*self.z1+sigma*self.z2, mu+tau*self.z1-sigma*self.z2

    @staticmethod
    def side(y, eta, dropout):
        """Log-probability of one side's count; gammaln accepts half-integer gap sizes."""
        logit = dropout[0]+dropout[1]*eta
        log_q, log_keep = -np.logaddexp(0, -logit), -np.logaddexp(0, logit)
        present = log_keep+y*eta-np.exp(eta)-gammaln(y+1)
        return np.where(y > 0, present, np.logaddexp(log_q, log_keep-np.exp(eta)))

    def pair_logpost(self, params, y1, y2):
        e1, e2 = self.grid(params)
        q = self.unpack(params)[3]
        y1, y2 = np.asarray(y1)[:, None, None], np.asarray(y2)[:, None, None]
        return self.logw+self.side(y1, e1, q)+self.side(y2, e2, q), e1, e2

    def negloglik(self, params):
        terms, e1, e2 = self.pair_logpost(params, self.y1, self.y2)
        joint = logsumexp(terms.reshape(len(self.y1), -1), axis=1)
        q = self.unpack(params)[3]
        absent = logsumexp((self.logw+self.side(0., e1, q)+self.side(0., e2, q)).ravel())
        return -(self.multiplicity*(joint-np.log1p(-np.exp(absent)))).sum()

    def posterior_counts(self, params, y1, y2):
        """Posterior mean of each side's latent rate given both observed counts."""
        terms, e1, e2 = self.pair_logpost(params, y1, y2)
        flat = terms.reshape(len(y1), -1)
        weights = np.exp(flat-logsumexp(flat, axis=1, keepdims=True))
        return weights@np.exp(e1).ravel(), weights@np.exp(e2).ravel()

    def single_posterior(self, params, y):
        """Posterior mean rate for an observed edge with no defined mirror."""
        mu, tau, sigma, dropout = self.unpack(params)
        z, w = np.polynomial.hermite_e.hermegauss(80)
        eta = mu+np.hypot(tau, sigma)*z
        y = np.asarray(y)[:, None]
        terms = np.log(w)+self.side(y, eta, dropout)
        weights = np.exp(terms-logsumexp(terms, axis=1, keepdims=True))
        return weights@np.exp(eta)

    def simulate(self, params, size, rng):
        """Draw censored pairs; used for the posterior predictive presence check."""
        mu, tau, sigma, dropout = self.unpack(params)
        out = []
        while sum(len(x) for x in out) < size:
            theta = rng.normal(mu, tau, size)
            eta = theta[:, None]+rng.normal(0, sigma, (size, 2))
            q = 1/(1+np.exp(-(dropout[0]+dropout[1]*eta)))
            y = rng.poisson(np.exp(eta))*(rng.random((size, 2)) >= q)
            out.append(y[y.sum(axis=1) > 0])
        return np.concatenate(out)[:size]


def analyse(name, counts, mirror, directed, nulls, rng):
    hit, total = presence_by_bin(counts, mirror, directed)
    null_hit = np.zeros((nulls, len(BINS)))
    edges, values = list(counts), list(counts.values())
    for k in range(nulls):
        rewired = dict(zip(rewire(edges, directed, rng), values))
        null_hit[k] = presence_by_bin(rewired, mirror, directed)[0]
    pairs = mirror_pairs(counts, mirror, directed)
    y1 = np.array([p[2] for p in pairs])
    y2 = np.array([p[3] for p in pairs])
    model = Model(y1, y2)
    start = np.array([np.log(np.mean(np.r_[y1, y2])+1e-9), 0., -1., -1., 0.])
    fit = minimize(model.negloglik, start, method='Nelder-Mead',
                   options={'xatol': 1e-6, 'fatol': 1e-6, 'maxiter': 8000})
    assert fit.success, fit.message
    params = fit.x
    _, tau, sigma, dropout = Model.unpack(params)
    tau, sigma = float(tau), float(sigma)
    sim = model.simulate(params, 200000, rng)
    sim_counts = np.concatenate([sim[:, 0], sim[:, 1]])
    sim_other = np.concatenate([sim[:, 1], sim[:, 0]])
    bins = []
    for k, (lo, hi) in enumerate(BINS):
        sel = (sim_counts >= lo) & (sim_counts < hi+1)
        chance = null_hit[:, k]/max(total[k], 1)
        bins.append({'counts': [lo, hi if hi < 10**9 else None], 'edges_with_mirror': int(total[k]),
                     'mirror_present': float(hit[k]/total[k]) if total[k] else None,
                     'rewired_chance_mean': float(chance.mean()),
                     'rewired_chance_95': [float(np.quantile(chance, .025)), float(np.quantile(chance, .975))],
                     'model_predicted_present': float(np.mean(sim_other[sel] > 0)) if sel.any() else None})
    post1, post2 = model.posterior_counts(params, y1, y2)
    shrunk = {}
    for (x, y, _, _), p1, p2 in zip(pairs, post1, post2):
        if x in counts:
            shrunk[x] = (float(p1), True, y in counts)
        if y in counts:
            shrunk[y] = (float(p2), True, x in counts)
    singles = [e for e in counts if e not in shrunk]
    for e, value in zip(singles, model.single_posterior(params, [counts[e] for e in singles])):
        shrunk[e] = (float(value), False, False)
    observed_both = (y1 > 0) & (y2 > 0)
    summary = {
        'edges': len(counts), 'mirror_pairs': len(pairs),
        'edges_without_defined_mirror': len(singles),
        'pairs_both_present': int(observed_both.sum()),
        'log1p_count_correlation_both_present': float(np.corrcoef(np.log1p(y1[observed_both]), np.log1p(y2[observed_both]))[0, 1]),
        'model': {'mu': float(params[0]), 'tau': tau, 'sigma': sigma,
                  'dropout_logit_intercept': float(dropout[0]), 'dropout_logit_slope': float(dropout[1]),
                  'latent_reliability': tau**2/(tau**2+sigma**2),
                  'negloglik': float(fit.fun), 'iterations': int(fit.nit)},
        'presence_by_count': bins,
        'rewired_graphs': nulls,
    }
    return summary, shrunk


def eb_table(counts, shrunk):
    rows = []
    for c in [1, 2, 3, 5, 10, 20]:
        confirmed = [shrunk[e][0] for e, v in counts.items() if v == c and shrunk[e][2]]
        orphan = [shrunk[e][0] for e, v in counts.items() if v == c and shrunk[e][1] and not shrunk[e][2]]
        rows.append({'observed': c,
                     'eb_mirror_present': float(np.median(confirmed)) if confirmed else None, 'n_mirror_present': len(confirmed),
                     'eb_mirror_absent': float(np.median(orphan)) if orphan else None, 'n_mirror_absent': len(orphan)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--graph', default='runs/c302-audit.json')
    p.add_argument('--nulls', type=int, default=200)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--summary', required=True)
    p.add_argument('--edges', required=True, help='per-edge empirical-Bayes output')
    a = p.parse_args()
    for path in [a.summary, a.edges]:
        assert not Path(path).exists(), 'outputs must be new'
    graph = json.loads(Path(a.graph).read_text())
    names = {x['id'] for x in graph['neurons']}
    mirror = mirror_map(names)
    chemical = {(e['pre'], e['post']): float(e['synapse_count']) for e in graph['chemical']}
    gaps = {key(e['a'], e['b'], False): float(e['size']) for e in graph['gaps']}
    assert len(chemical) == len(graph['chemical']) and len(gaps) == len(graph['gaps'])
    rng = np.random.default_rng(a.seed)
    report = {'format': 'wormsim-mirror-reliability', 'graph_sha256': digest(a.graph), 'seed': a.seed,
              'mirrored_neuron_pairs': sum(mirror[x] != x for x in names)//2,
              'unpaired_neurons': sum(mirror[x] == x for x in names)}
    per_edge = {'format': 'wormsim-edge-reliability', 'graph_sha256': report['graph_sha256'],
                'fields': ['eb_count', 'mirror_defined', 'mirror_present']}
    for name, counts, directed in [('chemical', chemical, True), ('gap', gaps, False)]:
        summary, shrunk = analyse(name, counts, mirror, directed, a.nulls, rng)
        summary['eb_by_observed_count'] = eb_table(counts, shrunk)
        report[name] = summary
        ordered = sorted(counts)
        per_edge[name] = [{'pair': list(e), 'count': counts[e], 'eb_count': round(shrunk[e][0], 6),
                           'mirror_defined': shrunk[e][1], 'mirror_present': shrunk[e][2]} for e in ordered]
    Path(a.summary).write_text(json.dumps(report, indent=1)+'\n')
    Path(a.edges).write_text(json.dumps(per_edge, separators=(',', ':'))+'\n')


if __name__ == '__main__':
    main()
