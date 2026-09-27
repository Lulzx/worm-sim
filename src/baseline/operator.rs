//! Exact matrix structure, selected at import time. No approximate sparsification.
use crate::Result;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum Operator {
    Zero {
        rows: usize,
        cols: usize,
    },
    Identity {
        size: usize,
    },
    LaggedDiagonal {
        size: usize,
        lags: usize,
        values: Vec<f64>,
    },
    Csr {
        rows: usize,
        cols: usize,
        offsets: Vec<usize>,
        columns: Vec<usize>,
        values: Vec<f64>,
    },
    Dense {
        rows: usize,
        cols: usize,
        values: Vec<f64>,
    },
}
impl Operator {
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::Zero { rows, cols }
            | Self::Csr { rows, cols, .. }
            | Self::Dense { rows, cols, .. } => (*rows, *cols),
            Self::Identity { size } => (*size, *size),
            Self::LaggedDiagonal { size, lags, .. } => (*size, size.saturating_mul(*lags)),
        }
    }
    pub fn validate(&self) -> Result<()> {
        let (rows, cols) = self.shape();
        if rows == 0
            || cols == 0
            || rows > 4096
            || cols > 1_000_000
            || rows.checked_mul(cols).is_none_or(|n| n > 16_000_000)
        {
            return Err("invalid operator dimensions".into());
        }
        match self {
            Self::Zero { .. } | Self::Identity { .. } => {}
            Self::LaggedDiagonal { values, .. } => {
                if values.len() != cols || values.iter().any(|v| !v.is_finite()) {
                    return Err("invalid diagonal values".into());
                }
            }
            Self::Dense { values, .. } => {
                if values.len() != rows * cols || values.iter().any(|v| !v.is_finite()) {
                    return Err("invalid dense values".into());
                }
            }
            Self::Csr {
                offsets,
                columns,
                values,
                ..
            } => {
                if offsets.len() != rows + 1
                    || offsets[0] != 0
                    || offsets[rows] != values.len()
                    || columns.len() != values.len()
                    || values.iter().any(|v| !v.is_finite())
                    || offsets.windows(2).any(|w| w[0] > w[1])
                {
                    return Err("invalid CSR storage".into());
                }
                for row in 0..rows {
                    let c = &columns[offsets[row]..offsets[row + 1]];
                    if c.iter().any(|&i| i >= cols) || c.windows(2).any(|w| w[0] >= w[1]) {
                        return Err("CSR columns must be bounded, sorted and unique".into());
                    }
                }
            }
        }
        Ok(())
    }
    /// Iterate nonzero/explicit coefficients; lagged diagonals never materialize.
    pub fn coefficients(&self, mut f: impl FnMut(usize, usize, f64)) {
        match self {
            Self::Zero { .. } => {}
            Self::Identity { size } => {
                for i in 0..*size {
                    f(i, i, 1.0);
                }
            }
            Self::LaggedDiagonal { size, lags, values } => {
                for lag in 0..*lags {
                    for row in 0..*size {
                        f(row, lag * size + row, values[lag * size + row]);
                    }
                }
            }
            Self::Csr {
                rows,
                offsets,
                columns,
                values,
                ..
            } => {
                for row in 0..*rows {
                    for e in offsets[row]..offsets[row + 1] {
                        f(row, columns[e], values[e]);
                    }
                }
            }
            Self::Dense { rows, cols, values } => {
                for row in 0..*rows {
                    for col in 0..*cols {
                        f(row, col, values[row * cols + col]);
                    }
                }
            }
        }
    }
    /// A times B, with B and output row-major and `batch` contiguous columns.
    /// Call validate once at an interchange boundary before using this kernel.
    pub fn multiply(&self, right: &[f64], batch: usize, out: &mut [f64]) -> Result<()> {
        let (rows, cols) = self.shape();
        if batch == 0
            || cols.checked_mul(batch) != Some(right.len())
            || rows.checked_mul(batch) != Some(out.len())
        {
            return Err("matrix product shape mismatch".into());
        }
        out.fill(0.0);
        self.coefficients(|row, col, value| {
            let source = &right[col * batch..(col + 1) * batch];
            let target = &mut out[row * batch..(row + 1) * batch];
            for (dst, &src) in target.iter_mut().zip(source) {
                *dst += value * src;
            }
        });
        Ok(())
    }
    pub fn add_column_block(&self, start: usize, width: usize, out: &mut [f64]) -> Result<()> {
        let (rows, cols) = self.shape();
        if rows.checked_mul(width) != Some(out.len())
            || start.checked_add(width).is_none_or(|end| end > cols)
        {
            return Err("column block out of bounds".into());
        }
        self.coefficients(|row, col, value| {
            if col >= start && col < start + width {
                out[row * width + col - start] += value;
            }
        });
        Ok(())
    }
    pub fn dense(&self) -> Vec<f64> {
        let (r, c) = self.shape();
        let mut out = vec![0.0; r * c];
        self.coefficients(|r, k, v| out[r * c + k] = v);
        out
    }
    pub fn scalar_count(&self) -> usize {
        match self {
            Self::Zero { .. } | Self::Identity { .. } => 0,
            Self::Csr { values, .. }
            | Self::Dense { values, .. }
            | Self::LaggedDiagonal { values, .. } => values.len(),
        }
    }
}
