//! WST1: independent 256-frame lossless chunks. A presence bitmap distinguishes
//! missing data from zero; per-channel temporal XOR preserves every finite f64 bit.
//! Independent chunks support bounded-memory decoding and selective time windows.
use crate::Result;
use sha2::{Digest, Sha256};
use std::io::Read;
const CHUNK_ROWS: usize = 256;
const MAX_VALUES: usize = 16_000_000;
#[derive(Debug)]
pub struct Matrix {
    pub rows: usize,
    pub columns: usize,
    pub values: Vec<Option<f64>>,
}
fn put_varint(mut value: u64, out: &mut Vec<u8>) {
    while value >= 128 {
        out.push(value as u8 | 128);
        value >>= 7;
    }
    out.push(value as u8);
}
fn get_varint(bytes: &[u8], pos: &mut usize) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        let b = *bytes.get(*pos).ok_or("truncated trace varint")?;
        *pos += 1;
        if shift == 63 && b > 1 {
            return Err("trace varint overflow".into());
        }
        value |= ((b & 127) as u64) << shift;
        if b < 128 {
            return Ok(value);
        }
    }
    Err("invalid trace varint".into())
}
fn take<'a>(bytes: &'a [u8], pos: &mut usize, len: usize) -> Result<&'a [u8]> {
    let end = pos.checked_add(len).ok_or("trace offset overflow")?;
    let slice = bytes.get(*pos..end).ok_or("truncated trace chunk")?;
    *pos = end;
    Ok(slice)
}
fn u32_at(bytes: &[u8], pos: &mut usize) -> Result<usize> {
    Ok(u32::from_le_bytes(take(bytes, pos, 4)?.try_into().unwrap()) as usize)
}
fn dimensions(rows: usize, cols: usize) -> Result<()> {
    if cols == 0 || rows == 0 || rows.checked_mul(cols).is_none_or(|n| n > MAX_VALUES) {
        return Err("invalid trace dimensions or size limit exceeded".into());
    }
    Ok(())
}
fn chunk_digest(rows: usize, cols: usize, start: usize, payload: &[u8]) -> Vec<u8> {
    let mut hash = Sha256::new();
    hash.update(b"WST1");
    for value in [rows, cols, start] {
        hash.update((value as u32).to_le_bytes());
    }
    hash.update(payload);
    hash.finalize().to_vec()
}
pub fn encode(matrix: &Matrix) -> Result<Vec<u8>> {
    dimensions(matrix.rows, matrix.columns)?;
    if matrix.values.len() != matrix.rows * matrix.columns
        || matrix.values.iter().flatten().any(|v| !v.is_finite())
    {
        return Err("trace shape mismatch or nonfinite value".into());
    }
    let mut out = b"WST1".to_vec();
    out.extend((matrix.rows as u32).to_le_bytes());
    out.extend((matrix.columns as u32).to_le_bytes());
    for start in (0..matrix.rows).step_by(CHUNK_ROWS) {
        let rows = CHUNK_ROWS.min(matrix.rows - start);
        let count = rows * matrix.columns;
        let mut payload = vec![0u8; count.div_ceil(8)];
        for col in 0..matrix.columns {
            let mut previous = 0u64;
            for row in 0..rows {
                let bit = col * rows + row;
                if let Some(value) = matrix.values[(start + row) * matrix.columns + col] {
                    payload[bit / 8] |= 1 << (bit % 8);
                    let bits = value.to_bits();
                    put_varint(bits ^ previous, &mut payload);
                    previous = bits;
                }
            }
        }
        let compressed =
            zstd::stream::encode_all(payload.as_slice(), 3).map_err(|e| e.to_string())?;
        out.extend((compressed.len() as u32).to_le_bytes());
        out.extend(chunk_digest(matrix.rows, matrix.columns, start, &payload));
        out.extend(compressed);
    }
    Ok(out)
}
/// Decode just the requested rows. All chunk framing is checked, while payload
/// checksums are verified only for chunks actually read. Empty ranges are invalid.
pub fn decode_range(bytes: &[u8], start: usize, end: usize) -> Result<Matrix> {
    if bytes.get(..4) != Some(b"WST1") {
        return Err("invalid WST1 header".into());
    }
    let mut pos = 4;
    let rows = u32_at(bytes, &mut pos)?;
    let cols = u32_at(bytes, &mut pos)?;
    dimensions(rows, cols)?;
    if start >= end || end > rows {
        return Err("invalid trace row range".into());
    }
    let mut values = vec![None; (end - start) * cols];
    for chunk_start in (0..rows).step_by(CHUNK_ROWS) {
        let length = u32_at(bytes, &mut pos)?;
        let expected_hash = take(bytes, &mut pos, 32)?;
        let compressed = take(bytes, &mut pos, length)?;
        let chunk_rows = CHUNK_ROWS.min(rows - chunk_start);
        let chunk_end = chunk_start + chunk_rows;
        if chunk_end <= start || chunk_start >= end {
            continue;
        }
        let count = chunk_rows * cols;
        let mask_len = count.div_ceil(8);
        let max_payload = mask_len + count * 10;
        let mut payload = Vec::new();
        zstd::stream::read::Decoder::new(compressed)
            .map_err(|e| e.to_string())?
            .take(max_payload as u64 + 1)
            .read_to_end(&mut payload)
            .map_err(|e| e.to_string())?;
        if payload.len() > max_payload
            || payload.len() < mask_len
            || chunk_digest(rows, cols, chunk_start, &payload)[..] != *expected_hash
        {
            return Err("trace payload checksum or size mismatch".into());
        }
        let mut cursor = mask_len;
        for col in 0..cols {
            let mut previous = 0u64;
            for row in 0..chunk_rows {
                let bit = col * chunk_rows + row;
                if payload[bit / 8] & (1 << (bit % 8)) != 0 {
                    previous ^= get_varint(&payload, &mut cursor)?;
                    let value = f64::from_bits(previous);
                    if !value.is_finite() {
                        return Err("nonfinite trace value".into());
                    }
                    let absolute = chunk_start + row;
                    if absolute >= start && absolute < end {
                        values[(absolute - start) * cols + col] = Some(value);
                    }
                }
            }
        }
        if cursor != payload.len() {
            return Err("trailing trace payload".into());
        }
    }
    if pos != bytes.len() {
        return Err("trailing archive bytes".into());
    }
    Ok(Matrix {
        rows: end - start,
        columns: cols,
        values,
    })
}
