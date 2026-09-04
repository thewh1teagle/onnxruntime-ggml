//! Numpy-style broadcasting helpers shared by the host evaluator and the ggml
//! emitter (which needs the result shape to decide who repeats into whom).

use crate::error::{Error, Result};
use crate::host::tensor::strides_of;

/// The broadcast result of two shapes, or an error naming both.
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Result<Vec<usize>> {
    let rank = a.len().max(b.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        let da = if i < rank - a.len() { 1 } else { a[i - (rank - a.len())] };
        let db = if i < rank - b.len() { 1 } else { b[i - (rank - b.len())] };
        out[i] = if da == db {
            da
        } else if da == 1 {
            db
        } else if db == 1 {
            da
        } else {
            return Err(Error::shape(format!("cannot broadcast {a:?} with {b:?}")));
        };
    }
    Ok(out)
}

/// Broadcast of any number of shapes.
pub fn broadcast_all(shapes: &[&[usize]]) -> Result<Vec<usize>> {
    let mut out: Vec<usize> = Vec::new();
    for s in shapes {
        out = broadcast_shapes(&out, s)?;
    }
    Ok(out)
}

/// For each element of `to` (row-major), the flat index into a tensor of
/// shape `from` that broadcasts to it.
pub fn broadcast_index(from: &[usize], to: &[usize]) -> Vec<usize> {
    let rank = to.len();
    let offset = rank - from.len();
    let from_strides = strides_of(from);
    // stride 0 on broadcast dimensions
    let mut eff = vec![0usize; rank];
    for i in 0..from.len() {
        eff[i + offset] = if from[i] == 1 { 0 } else { from_strides[i] };
    }
    let n: usize = to.iter().product();
    let mut out = Vec::with_capacity(n);
    let mut idx = vec![0usize; rank];
    for _ in 0..n {
        let mut flat = 0;
        for d in 0..rank {
            flat += idx[d] * eff[d];
        }
        out.push(flat);
        // increment the multi-index
        for d in (0..rank).rev() {
            idx[d] += 1;
            if idx[d] < to[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    out
}

/// Whether `from` can broadcast into `to` without `to` changing.
pub fn broadcasts_into(from: &[usize], to: &[usize]) -> bool {
    matches!(broadcast_shapes(from, to), Ok(s) if s == to)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shapes() {
        assert_eq!(broadcast_shapes(&[1, 3], &[2, 1]).unwrap(), vec![2, 3]);
        assert_eq!(broadcast_shapes(&[], &[4]).unwrap(), vec![4]);
        assert_eq!(broadcast_shapes(&[5, 1, 4], &[3, 1]).unwrap(), vec![5, 3, 4]);
        assert!(broadcast_shapes(&[2], &[3]).is_err());
    }

    #[test]
    fn index_map() {
        assert_eq!(broadcast_index(&[3], &[2, 3]), vec![0, 1, 2, 0, 1, 2]);
        assert_eq!(broadcast_index(&[2, 1], &[2, 3]), vec![0, 0, 0, 1, 1, 1]);
        assert_eq!(broadcast_index(&[], &[2, 2]), vec![0, 0, 0, 0]);
    }
}
