//! WSC1: lossless graph-specific storage. Canonical edge keys are delta-varints;
//! numeric columns use XOR-varints; repeated metadata is dictionary encoded.
//! Zstd compresses the resulting payload. SHA-256 detects payload corruption.
use crate::{
    Result,
    data::{ChemicalEdge, GapEdge, Graph, IndexedGraph, Neuron, Provenance},
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Read};
const MAX_PAYLOAD: usize = 64 * 1024 * 1024;
#[derive(Serialize, Deserialize)]
struct Metadata {
    neurons: Vec<Neuron>,
    dictionary: Vec<EdgeMetadata>,
}
#[derive(Serialize, Deserialize)]
struct EdgeMetadata {
    provenance: Vec<Provenance>,
    receptor_candidates: Vec<String>,
}
fn varint(mut x: u64, out: &mut Vec<u8>) {
    while x >= 128 {
        out.push((x as u8 & 127) | 128);
        x >>= 7;
    }
    out.push(x as u8);
}
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.pos.checked_add(n).ok_or("length overflow")?;
        let s = self
            .bytes
            .get(self.pos..end)
            .ok_or("truncated WSC1 payload")?;
        self.pos = end;
        Ok(s)
    }
    fn varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..70).step_by(7) {
            let b = self.take(1)?[0];
            if shift == 63 && b > 1 {
                return Err("varint overflow".into());
            }
            value |= ((b & 127) as u64) << shift;
            if b < 128 {
                return Ok(value);
            }
        }
        Err("invalid varint".into())
    }
    fn count(&mut self) -> Result<usize> {
        let n = usize::try_from(self.varint()?).map_err(|_| "count overflow")?;
        if n > MAX_PAYLOAD {
            return Err("count exceeds codec limit".into());
        }
        Ok(n)
    }
}
pub fn encode(graph: &IndexedGraph) -> Result<Vec<u8>> {
    let mut dictionary = Vec::new();
    let mut lookup = BTreeMap::<Vec<u8>, u64>::new();
    let mut ids = Vec::new();
    for (provenance, receptors) in graph
        .graph
        .chemical
        .iter()
        .map(|e| (&e.provenance, e.receptor_candidates.clone()))
        .chain(graph.graph.gaps.iter().map(|e| (&e.provenance, Vec::new())))
    {
        let metadata = EdgeMetadata {
            provenance: provenance.clone(),
            receptor_candidates: receptors,
        };
        let key = serde_json::to_vec(&metadata).map_err(|e| e.to_string())?;
        let id = if let Some(&i) = lookup.get(&key) {
            i
        } else {
            let i = dictionary.len() as u64;
            lookup.insert(key, i);
            dictionary.push(metadata);
            i
        };
        ids.push(id);
    }
    let meta = serde_json::to_vec(&Metadata {
        neurons: graph.graph.neurons.clone(),
        dictionary,
    })
    .map_err(|e| e.to_string())?;
    let mut payload = Vec::new();
    varint(meta.len() as u64, &mut payload);
    payload.extend(meta);
    varint(graph.chemical.len() as u64, &mut payload);
    varint(graph.gaps.len() as u64, &mut payload);
    let n = graph.names.len() as u64;
    let mut previous = 0;
    let mut count_bits = 0;
    let mut sign_bits = 0;
    for (i, &(a, b, count, sign)) in graph.chemical.iter().enumerate() {
        let key = a as u64 * n + b as u64;
        varint(key - previous, &mut payload);
        previous = key;
        varint(count.to_bits() ^ count_bits, &mut payload);
        count_bits = count.to_bits();
        varint(sign.to_bits() ^ sign_bits, &mut payload);
        sign_bits = sign.to_bits();
        varint(ids[i], &mut payload);
    }
    previous = 0;
    count_bits = 0;
    for (i, &(a, b, size)) in graph.gaps.iter().enumerate() {
        let key = a as u64 * n + b as u64;
        varint(key - previous, &mut payload);
        previous = key;
        varint(size.to_bits() ^ count_bits, &mut payload);
        count_bits = size.to_bits();
        varint(ids[graph.chemical.len() + i], &mut payload);
    }
    if payload.len() > MAX_PAYLOAD {
        return Err("graph exceeds codec limit".into());
    }
    let mut output = b"WSC1".to_vec();
    output.extend((payload.len() as u64).to_le_bytes());
    output.extend(Sha256::digest(&payload));
    output.extend(zstd::stream::encode_all(payload.as_slice(), 3).map_err(|e| e.to_string())?);
    Ok(output)
}
pub fn decode(bytes: &[u8]) -> Result<IndexedGraph> {
    if bytes.len() < 44 || &bytes[..4] != b"WSC1" {
        return Err("invalid WSC1 header".into());
    }
    let size = u64::from_le_bytes(bytes[4..12].try_into().unwrap());
    if size > MAX_PAYLOAD as u64 {
        return Err("decoded payload exceeds codec limit".into());
    }
    let decoder = zstd::stream::read::Decoder::new(&bytes[44..]).map_err(|e| e.to_string())?;
    let mut payload = Vec::new();
    decoder
        .take(size + 1)
        .read_to_end(&mut payload)
        .map_err(|e| e.to_string())?;
    if payload.len() as u64 != size || Sha256::digest(&payload)[..] != bytes[12..44] {
        return Err("WSC1 size or checksum mismatch".into());
    }
    let mut r = Reader {
        bytes: &payload,
        pos: 0,
    };
    let length = r.count()?;
    let meta: Metadata = serde_json::from_slice(r.take(length)?).map_err(|e| e.to_string())?;
    let n = meta.neurons.len() as u64;
    if n == 0 || n > 65535 {
        return Err("invalid neuron count".into());
    }
    let chemical_count = r.count()?;
    let gap_count = r.count()?;
    if chemical_count > payload.len() / 4 || gap_count > payload.len() / 3 {
        return Err("invalid edge counts".into());
    }
    let mut chemical = Vec::new();
    let mut gaps = Vec::new();
    let mut previous = 0u64;
    let mut count_bits = 0u64;
    let mut sign_bits = 0u64;
    for _ in 0..chemical_count {
        let key = previous
            .checked_add(r.varint()?)
            .ok_or("edge key overflow")?;
        previous = key;
        if key >= n * n {
            return Err("edge index out of range".into());
        }
        count_bits ^= r.varint()?;
        sign_bits ^= r.varint()?;
        let metadata = meta
            .dictionary
            .get(r.count()?)
            .ok_or("metadata ID out of range")?;
        chemical.push(ChemicalEdge {
            pre: meta.neurons[(key / n) as usize].id.clone(),
            post: meta.neurons[(key % n) as usize].id.clone(),
            synapse_count: f64::from_bits(count_bits),
            sign_prior: f64::from_bits(sign_bits),
            provenance: metadata.provenance.clone(),
            receptor_candidates: metadata.receptor_candidates.clone(),
        });
    }
    previous = 0;
    count_bits = 0;
    for _ in 0..gap_count {
        let key = previous
            .checked_add(r.varint()?)
            .ok_or("edge key overflow")?;
        previous = key;
        if key >= n * n {
            return Err("edge index out of range".into());
        }
        count_bits ^= r.varint()?;
        let metadata = meta
            .dictionary
            .get(r.count()?)
            .ok_or("metadata ID out of range")?;
        gaps.push(GapEdge {
            a: meta.neurons[(key / n) as usize].id.clone(),
            b: meta.neurons[(key % n) as usize].id.clone(),
            size: f64::from_bits(count_bits),
            provenance: metadata.provenance.clone(),
        });
    }
    if r.pos != payload.len() {
        return Err("trailing payload bytes".into());
    }
    Graph {
        schema_version: 1,
        neurons: meta.neurons,
        chemical,
        gaps,
    }
    .compile()
}
