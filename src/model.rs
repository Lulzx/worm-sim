//! Structure-of-arrays Level 0 kernel. Edges use compact u16 neuron indices.
use crate::{
    Result,
    data::IndexedGraph,
    math::{Scalar, inverse_softplus},
};

#[derive(Clone, Debug)]
pub struct Model {
    pub graph: IndexedGraph,
    pub pre: Vec<u16>,
    pub post: Vec<u16>,
    pub counts: Vec<f64>,
    pub incoming_offsets: Vec<usize>,
    pub parameter_edge: Vec<usize>,
    pub gap_a: Vec<u16>,
    pub gap_b: Vec<u16>,
    pub gap_sizes: Vec<f64>,
}
/// Unconstrained groups: tau, rest, threshold, slope, calcium_tau, calcium_scale,
/// chemical_strength, chemical_sign, gap_strength, synapse_tau.
#[derive(Clone, Debug)]
pub struct Parameters<S> {
    pub raw: Vec<S>,
}
#[derive(Clone, Debug)]
pub struct Prepared<S> {
    pub inv_tau: Vec<S>,
    pub rest: Vec<S>,
    pub threshold: Vec<S>,
    pub slope: Vec<S>,
    pub inv_calcium_tau: Vec<S>,
    pub calcium_scale: Vec<S>,
    pub weight: Vec<S>,
    pub reversal: Vec<S>,
    pub gap: Vec<S>,
    pub inv_synapse_tau: S,
}
impl Model {
    pub fn new(graph: IndexedGraph) -> Result<Self> {
        if graph.names.len() > u16::MAX as usize {
            return Err("compact kernel supports at most 65535 neurons".into());
        }
        let mut order: Vec<_> = (0..graph.chemical.len()).collect();
        order.sort_by_key(|&e| (graph.chemical[e].1, graph.chemical[e].0));
        let mut incoming_offsets = vec![0; graph.names.len() + 1];
        for &e in &order {
            incoming_offsets[graph.chemical[e].1 + 1] += 1;
        }
        for i in 1..incoming_offsets.len() {
            incoming_offsets[i] += incoming_offsets[i - 1];
        }
        Ok(Self {
            pre: order.iter().map(|&e| graph.chemical[e].0 as u16).collect(),
            post: order.iter().map(|&e| graph.chemical[e].1 as u16).collect(),
            counts: order.iter().map(|&e| graph.chemical[e].2).collect(),
            incoming_offsets,
            parameter_edge: order,
            gap_a: graph.gaps.iter().map(|e| e.0 as u16).collect(),
            gap_b: graph.gaps.iter().map(|e| e.1 as u16).collect(),
            gap_sizes: graph.gaps.iter().map(|e| e.2).collect(),
            graph,
        })
    }
    pub fn n(&self) -> usize {
        self.graph.names.len()
    }
    pub fn state_len(&self) -> usize {
        3 * self.n()
    }
    pub fn parameter_count(&self) -> usize {
        6 * self.n() + 2 * self.pre.len() + self.gap_a.len() + 1
    }
    pub fn defaults(&self) -> Parameters<f64> {
        let n = self.n();
        let mut raw = Vec::with_capacity(self.parameter_count());
        for value in [
            inverse_softplus(0.1),
            -0.5,
            0.0,
            inverse_softplus(4.0),
            inverse_softplus(0.5),
            inverse_softplus(1.0),
        ] {
            raw.extend(std::iter::repeat_n(value, n));
        }
        raw.extend(std::iter::repeat_n(inverse_softplus(0.02), self.pre.len()));
        raw.extend(self.graph.chemical.iter().map(|e| {
            let p = e.3.clamp(1e-6, 1.0 - 1e-6);
            (p / (1.0 - p)).ln()
        }));
        raw.extend(std::iter::repeat_n(
            inverse_softplus(0.02),
            self.gap_a.len(),
        ));
        raw.push(inverse_softplus(0.02));
        Parameters { raw }
    }
    pub fn prepare<S: Scalar>(&self, p: &Parameters<S>) -> Result<Prepared<S>> {
        if p.raw.len() != self.parameter_count() || p.raw.iter().any(|x| !x.value().is_finite()) {
            return Err("parameter count mismatch or nonfinite value".into());
        }
        let n = self.n();
        let m = self.pre.len();
        let one = S::constant(1.0);
        let positive = |i: usize| p.raw[i].softplus() + S::constant(1e-9);
        Ok(Prepared {
            inv_tau: (0..n).map(|i| one / positive(i)).collect(),
            rest: p.raw[n..2 * n].to_vec(),
            threshold: p.raw[2 * n..3 * n].to_vec(),
            slope: (3 * n..4 * n).map(positive).collect(),
            inv_calcium_tau: (4 * n..5 * n).map(|i| one / positive(i)).collect(),
            calcium_scale: (5 * n..6 * n).map(positive).collect(),
            weight: (0..m)
                .map(|i| positive(6 * n + self.parameter_edge[i]) * S::constant(self.counts[i]))
                .collect(),
            reversal: (0..m)
                .map(|i| {
                    S::constant(2.0) * p.raw[6 * n + m + self.parameter_edge[i]].sigmoid() - one
                })
                .collect(),
            gap: (0..self.gap_a.len())
                .map(|i| positive(6 * n + 2 * m + i) * S::constant(self.gap_sizes[i]))
                .collect(),
            inv_synapse_tau: one / positive(self.parameter_count() - 1),
        })
    }
    pub fn initial<S: Scalar>(&self, p: &Prepared<S>) -> Vec<S> {
        let n = self.n();
        let mut y = vec![S::constant(0.0); self.state_len()];
        y[..n].copy_from_slice(&p.rest);
        for i in 0..n {
            y[n + i] = ((p.rest[i] - p.threshold[i]) * p.slope[i]).sigmoid();
        }
        for i in 0..n {
            let r = y[n + i];
            y[2 * n + i] = r / (S::constant(1.0) + r);
        }
        y
    }
    /// Pure RHS; caller owns all scratch. Each gap is evaluated once and applied
    /// with equal/opposite current. Silencing blocks release; ablation removes edges.
    pub fn rhs<S: Scalar>(
        &self,
        p: &Prepared<S>,
        y: &[S],
        input: &Inputs,
        release: &mut [S],
        dy: &mut [S],
    ) {
        let n = self.n();
        let zero = S::constant(0.0);
        let one = S::constant(1.0);
        for i in 0..n {
            release[i] = if input.silenced[i] || input.ablated[i] {
                zero
            } else {
                ((y[i] - p.threshold[i]) * p.slope[i]).sigmoid()
            };
            dy[i] = -(y[i] - p.rest[i]) + S::constant(input.current[i]);
            if input.conductance[i] != 0.0 {
                dy[i] = dy[i] + S::constant(input.conductance_drive[i])
                    - S::constant(input.conductance[i]) * y[i];
            }
            dy[n + i] = (release[i] - y[n + i]) * p.inv_calcium_tau[i];
            let s = y[2 * n + i];
            dy[2 * n + i] = (release[i] * (one - s) - s) * p.inv_synapse_tau;
        }
        // Reuse release scratch for effective presynaptic gates after all
        // gate/calcium derivatives are calculated. Mask once per neuron.
        for i in 0..n {
            release[i] = if input.silenced[i] || input.ablated[i] {
                zero
            } else {
                y[2 * n + i]
            };
        }
        for b in 0..n {
            if input.ablated[b] {
                continue;
            }
            let voltage = y[b];
            let mut current = zero;
            let start = self.incoming_offsets[b];
            let end = self.incoming_offsets[b + 1];
            for ((&a, &weight), &reversal) in self.pre[start..end]
                .iter()
                .zip(&p.weight[start..end])
                .zip(&p.reversal[start..end])
            {
                current = current + weight * release[a as usize] * (reversal - voltage);
            }
            dy[b] = dy[b] + current;
        }
        for e in 0..self.gap_a.len() {
            let a = self.gap_a[e] as usize;
            let b = self.gap_b[e] as usize;
            if !input.ablated[a] && !input.ablated[b] {
                let current = p.gap[e] * (y[b] - y[a]);
                dy[a] = dy[a] + current;
                dy[b] = dy[b] - current;
            }
        }
        for (i, derivative) in dy.iter_mut().enumerate().take(n) {
            *derivative = if input.ablated[i] {
                zero
            } else {
                *derivative * p.inv_tau[i]
            };
        }
    }
}
#[derive(Clone, Debug)]
pub struct Inputs {
    pub current: Vec<f64>,
    /// Sum of externally applied nonnegative conductances, in leak units.
    pub conductance: Vec<f64>,
    /// Sum of conductance times reversal potential for overlapping inputs.
    pub conductance_drive: Vec<f64>,
    pub silenced: Vec<bool>,
    pub ablated: Vec<bool>,
}
impl Inputs {
    pub fn new(n: usize) -> Self {
        Self {
            current: vec![0.0; n],
            conductance: vec![0.0; n],
            conductance_drive: vec![0.0; n],
            silenced: vec![false; n],
            ablated: vec![false; n],
        }
    }
}
