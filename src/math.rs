//! Forward-mode automatic differentiation. A tangent is one directional derivative.
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Scalar:
    Copy
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    fn constant(x: f64) -> Self;
    fn value(self) -> f64;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn sigmoid(self) -> Self {
        let one = Self::constant(1.0);
        if self.value() >= 0.0 {
            one / (one + (-self).exp())
        } else {
            let e = self.exp();
            e / (one + e)
        }
    }
    fn softplus(self) -> Self {
        let one = Self::constant(1.0);
        if self.value() > 0.0 {
            self + (one + (-self).exp()).ln()
        } else {
            (one + self.exp()).ln()
        }
    }
}
impl Scalar for f64 {
    fn constant(x: f64) -> Self {
        x
    }
    fn value(self) -> f64 {
        self
    }
    fn exp(self) -> Self {
        f64::exp(self)
    }
    fn ln(self) -> Self {
        f64::ln(self)
    }
}
#[derive(Clone, Copy, Debug)]
pub struct Dual {
    pub value: f64,
    pub tangent: f64,
}
impl Scalar for Dual {
    fn constant(value: f64) -> Self {
        Self {
            value,
            tangent: 0.0,
        }
    }
    fn value(self) -> f64 {
        self.value
    }
    fn exp(self) -> Self {
        let value = self.value.exp();
        Self {
            value,
            tangent: value * self.tangent,
        }
    }
    fn ln(self) -> Self {
        Self {
            value: self.value.ln(),
            tangent: self.tangent / self.value,
        }
    }
}
impl Add for Dual {
    type Output = Self;
    fn add(self, b: Self) -> Self {
        Self {
            value: self.value + b.value,
            tangent: self.tangent + b.tangent,
        }
    }
}
impl Sub for Dual {
    type Output = Self;
    fn sub(self, b: Self) -> Self {
        Self {
            value: self.value - b.value,
            tangent: self.tangent - b.tangent,
        }
    }
}
impl Mul for Dual {
    type Output = Self;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn mul(self, b: Self) -> Self {
        Self {
            value: self.value * b.value,
            tangent: self.tangent * b.value + self.value * b.tangent,
        }
    }
}
impl Div for Dual {
    type Output = Self;
    fn div(self, b: Self) -> Self {
        Self {
            value: self.value / b.value,
            tangent: (self.tangent * b.value - self.value * b.tangent) / (b.value * b.value),
        }
    }
}
impl Neg for Dual {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            value: -self.value,
            tangent: -self.tangent,
        }
    }
}
pub fn inverse_softplus(x: f64) -> f64 {
    if x > 30.0 { x } else { x.exp_m1().ln() }
}
