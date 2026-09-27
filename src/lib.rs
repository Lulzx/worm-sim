//! Differentiable CPU reference implementation of WormSim's Level 0 model.
//! Time is in seconds; voltage and current use consistent normalized units.
pub mod data;
pub mod fit;
pub mod math;
pub mod model;
pub mod solve;

pub type Result<T> = std::result::Result<T, String>;

pub mod codec;

pub mod fixtures;

pub mod import;

pub mod trace_codec;

pub mod baseline;

pub mod bench;

pub mod recordings;

pub mod initial_state;

pub mod parameters;
