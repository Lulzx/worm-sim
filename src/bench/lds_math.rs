//! Small row-major dense matrices for latent Gaussian inference.
use crate::Result;
pub(super) fn eye(n: usize, scale: f64) -> Vec<f64> {
    let mut a = vec![0.0; n * n];
    for i in 0..n {
        a[i * n + i] = scale;
    }
    a
}
pub(super) fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; a.len()];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = a[i * cols + j];
        }
    }
    out
}
pub(super) fn mm(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for x in 0..k {
            let v = a[i * k + x];
            for j in 0..n {
                c[i * n + j] += v * b[x * n + j];
            }
        }
    }
    c
}
pub(super) fn mv(a: &[f64], v: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    a.chunks_exact(cols)
        .take(rows)
        .map(|r| r.iter().zip(v).map(|(x, y)| x * y).sum())
        .collect()
}
pub(super) fn symmetrize(a: &mut [f64], n: usize) {
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (a[i * n + j] + a[j * n + i]);
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
    }
}
pub(super) fn chol(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut v = a[i * n + j];
            for k in 0..j {
                v -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !v.is_finite() || v <= 0.0 {
                    return Err("LDS covariance/normal matrix is not positive definite".into());
                }
                l[i * n + j] = v.sqrt();
            } else {
                l[i * n + j] = v / l[j * n + j];
            }
        }
    }
    Ok(l)
}
pub(super) fn solve(l: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut x = b.to_vec();
    for i in 0..n {
        for j in 0..i {
            x[i] -= l[i * n + j] * x[j];
        }
        x[i] /= l[i * n + i];
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            x[i] -= l[j * n + i] * x[j];
        }
        x[i] /= l[i * n + i];
    }
    x
}
pub(super) fn inverse(a: &[f64], n: usize) -> Result<Vec<f64>> {
    let l = chol(a, n)?;
    let mut out = vec![0.0; n * n];
    for j in 0..n {
        let mut b = vec![0.0; n];
        b[j] = 1.0;
        let x = solve(&l, &b, n);
        for i in 0..n {
            out[i * n + j] = x[i];
        }
    }
    symmetrize(&mut out, n);
    Ok(out)
}
/// Jacobi rotations of a symmetric matrix, with eigenvectors in columns.
/// The returned residual matrix also permits a Gershgorin eigenvalue bound.
pub(super) fn eigen(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut d = a.to_vec();
    let mut v = eye(n, 1.0);
    symmetrize(&mut d, n);
    for _ in 0..100 {
        let scale = (0..n).map(|i| d[i * n + i].abs()).fold(1.0, f64::max);
        let mut max: f64 = 0.0;
        for p in 0..n {
            for q in p + 1..n {
                let b = d[p * n + q];
                max = max.max(b.abs());
                if b.abs() < 1e-13 * scale {
                    continue;
                }
                let tau = (d[q * n + q] - d[p * n + p]) / (2.0 * b);
                let t = tau.signum() / (tau.abs() + (1.0 + tau * tau).sqrt());
                let t = if tau == 0.0 { 1.0 } else { t };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                let app = d[p * n + p];
                let aqq = d[q * n + q];
                d[p * n + p] = app - t * b;
                d[q * n + q] = aqq + t * b;
                d[p * n + q] = 0.0;
                d[q * n + p] = 0.0;
                for k in 0..n {
                    if k != p && k != q {
                        let x = d[k * n + p];
                        let y = d[k * n + q];
                        d[k * n + p] = c * x - s * y;
                        d[p * n + k] = d[k * n + p];
                        d[k * n + q] = s * x + c * y;
                        d[q * n + k] = d[k * n + q];
                    }
                    let x = v[k * n + p];
                    let y = v[k * n + q];
                    v[k * n + p] = c * x - s * y;
                    v[k * n + q] = s * x + c * y;
                }
            }
        }
        if max < 1e-12 * scale {
            return Ok((d, v));
        }
    }
    Err("symmetric eigensolver did not converge".into())
}
pub(super) fn norm_bound(a: &[f64], n: usize) -> Result<f64> {
    let gram = mm(&transpose(a, n, n), a, n, n, n);
    let (d, _) = eigen(&gram, n)?;
    let bound = (0..n)
        .map(|i| {
            d[i * n + i]
                + (0..n)
                    .filter(|&j| j != i)
                    .map(|j| d[i * n + j].abs())
                    .sum::<f64>()
        })
        .fold(0.0, f64::max);
    Ok((bound.max(0.0) * (1.0 + 1e-12) + 1e-15).sqrt())
}
pub(super) fn stabilize(a: &mut [f64], n: usize, cap: f64) -> Result<f64> {
    let before = norm_bound(a, n)?;
    if before > cap {
        for v in a {
            *v *= cap / before;
        }
    }
    Ok(before)
}
