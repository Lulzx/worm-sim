//! Explicit, deterministic learning-rate schedules for population fitting.
use crate::Result;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Optimizer {
    Adam {},
    /// Decoupled decay of every trainable raw coordinate, including kernel,
    /// classifier and observation log-gain. Frozen coordinates never decay.
    #[serde(rename = "adamw")]
    AdamW {
        weight_decay: f64,
    },
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::Adam {}
    }
}

impl Optimizer {
    pub fn validate(&self) -> Result<()> {
        if let Self::AdamW { weight_decay } = self
            && (!weight_decay.is_finite() || *weight_decay < 0.)
        {
            return Err("AdamW decay must be finite and nonnegative".into());
        }
        Ok(())
    }

    pub(crate) fn update(
        &self,
        adam: &mut super::population::Adam,
        values: &mut [f64],
        gradient: &[f64],
        trainable: &[bool],
        rate: f64,
    ) -> Result<()> {
        self.validate()?;
        if values.len() != gradient.len()
            || trainable.len() != values.len()
            || !rate.is_finite()
            || rate < 0.
            || values.iter().chain(gradient).any(|v| !v.is_finite())
        {
            return Err("invalid atlas optimizer inputs".into());
        }
        match self {
            Self::Adam {} => adam.update(values, gradient, rate),
            Self::AdamW { weight_decay } => {
                let decay = rate * weight_decay;
                if !decay.is_finite() || decay > 1. {
                    return Err("AdamW rate times decay must be at most one".into());
                }
                let before = values.to_vec();
                let gradient: Vec<_> = gradient
                    .iter()
                    .zip(trainable)
                    .map(|(g, active)| if *active { *g } else { 0. })
                    .collect();
                adam.update(values, &gradient, rate)?;
                for ((value, previous), active) in values.iter_mut().zip(before).zip(trainable) {
                    if *active {
                        *value -= decay * previous;
                    } else {
                        *value = previous;
                    }
                }
                if values.iter().any(|v| !v.is_finite()) {
                    return Err("nonfinite AdamW update".into());
                }
                Ok(())
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LearningRateSchedule {
    Constant {},
    /// First update uses the base rate; the last uses base * minimum_fraction.
    /// A one-update fit uses the base rate, with no decay interval.
    Cosine {
        minimum_fraction: f64,
    },
}

impl Default for LearningRateSchedule {
    fn default() -> Self {
        Self::Constant {}
    }
}

impl LearningRateSchedule {
    pub fn validate(&self) -> Result<()> {
        if let Self::Cosine { minimum_fraction } = self
            && (!minimum_fraction.is_finite() || !(0.0..=1.0).contains(minimum_fraction))
        {
            return Err("cosine minimum learning-rate fraction must be in [0,1]".into());
        }
        Ok(())
    }

    /// Updates are one-indexed; epoch zero is evaluation, not an optimizer step.
    pub fn rate(&self, base: f64, update: usize, total_updates: usize) -> Result<f64> {
        self.validate()?;
        if !base.is_finite() || base <= 0. || update == 0 || update > total_updates {
            return Err("invalid learning-rate schedule position or base rate".into());
        }
        let fraction = match self {
            Self::Constant {} => 1.,
            Self::Cosine { minimum_fraction } if total_updates > 1 => {
                let progress = (update - 1) as f64 / (total_updates - 1) as f64;
                minimum_fraction
                    + (1. - minimum_fraction) * 0.5 * (1. + (std::f64::consts::PI * progress).cos())
            }
            Self::Cosine { .. } => 1.,
        };
        Ok(base * fraction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bench::population::Adam;

    #[test]
    fn adamw_matches_independent_moments_and_keeps_frozen_coordinates() {
        let optimizer = Optimizer::AdamW { weight_decay: 0.2 };
        let mut adam = Adam::new(2);
        let mut values = [3., 7.];
        let (mut m, mut v, mut expected) = (0., 0., 3.);
        for (i, gradient) in [20_f64, -0.2, 1.].into_iter().enumerate() {
            let rate = [0.05, 0.03, 0.01][i];
            let g = gradient.clamp(-10., 10.);
            m = 0.9 * m + 0.1 * g;
            v = 0.999 * v + 0.001 * g * g;
            expected = (1. - rate * 0.2) * expected
                - rate * (m / (1. - 0.9_f64.powi(i as i32 + 1)))
                    / ((v / (1. - 0.999_f64.powi(i as i32 + 1))).sqrt() + 1e-8);
            optimizer
                .update(
                    &mut adam,
                    &mut values,
                    &[gradient, 1000.],
                    &[true, false],
                    rate,
                )
                .unwrap();
            assert!((values[0] - expected).abs() < 1e-13);
            assert_eq!(values[1], 7.);
        }
        // A zero data gradient still produces decay, without feeding decay
        // through Adam's moments or through gradient-norm clipping.
        let mut fresh = Adam::new(1);
        let mut value = [5.];
        optimizer
            .update(&mut fresh, &mut value, &[0.], &[true], 0.1)
            .unwrap();
        assert_eq!(value, [4.9]);
    }

    #[test]
    fn zero_decay_is_exact_legacy_adam_and_invalid_inputs_fail() {
        let mut legacy = Adam::new(2);
        let mut adamw = Adam::new(2);
        let (mut a, mut b) = ([2., -3.], [2., -3.]);
        for gradient in [[0.2, -0.1], [30., 40.], [-1., 2.]] {
            legacy.update(&mut a, &gradient, 0.01).unwrap();
            Optimizer::AdamW { weight_decay: 0. }
                .update(&mut adamw, &mut b, &gradient, &[true, true], 0.01)
                .unwrap();
            assert_eq!(a, b);
        }
        for decay in [-1., f64::NAN, f64::INFINITY] {
            assert!(
                Optimizer::AdamW {
                    weight_decay: decay
                }
                .validate()
                .is_err()
            );
        }
        assert!(
            Optimizer::AdamW { weight_decay: 2. }
                .update(&mut adamw, &mut b, &[1., 1.], &[true, true], 1.)
                .is_err()
        );
        assert!(
            Optimizer::AdamW { weight_decay: 0.1 }
                .update(&mut adamw, &mut b, &[1.], &[true, true], 0.01)
                .is_err()
        );
        assert!(serde_json::from_str::<Optimizer>(r#"{"kind":"adam","ignored":1}"#).is_err());
        assert!(
            serde_json::from_str::<Optimizer>(r#"{"kind":"adamw","weight_decay":0.1}"#).is_ok()
        );
    }
}
