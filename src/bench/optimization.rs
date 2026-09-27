//! Explicit, deterministic learning-rate schedules for population fitting.
use crate::Result;
use serde::{Deserialize, Serialize};

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
