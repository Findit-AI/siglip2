#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![deny(rust_2018_idioms, single_use_lifetimes, missing_docs)]

// Checkpoint-directory layout constants + the Apple-Silicon backend probe the
// auto-routing `from_dir` constructors share. Crate-internal; compiled wherever
// a `from_dir` constructor exists (the ONNX `from_dir` paths need only
// `inference` on a non-wasm host; the MLX probe inside is further target-gated).
#[cfg(all(feature = "inference", not(target_arch = "wasm32")))]
pub(crate) mod backend_select;
pub mod calibration;
pub mod embedding;
pub mod error;
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub mod image_enc;
pub mod image_view;
// MLX (`mlxrs`) inference backend — Apple Silicon only. Crate-internal:
// reached through the platform auto-routing in the `from_dir` constructors on
// `ImageEncoder` / `TextEncoder` / `Siglip2` (there is no public MLX entry point
// and no `mlx` feature — the backend is chosen by platform, see Cargo.toml).
// Compiled on `aarch64-apple-darwin` whenever the inference surface is built —
// it is part of that surface (it reuses the `tokenizers` text path and backs the
// `inference`-gated encoders), so an `--no-default-features` ONNX-free build
// pulls in neither it nor `mlxrs`'s runtime use. `mlxrs` binds the MLX C++
// runtime and has no other target, so the backend exists nowhere else.
#[cfg(all(feature = "inference", target_os = "macos", target_arch = "aarch64"))]
mod mlx;
pub mod options;
pub mod preproc;
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub(crate) mod session;
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub mod siglip2;
pub(crate) mod simd;
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub mod text_enc;

pub use calibration::Calibration;
pub use embedding::{Embedding, LabeledScore, LabeledScoreOwned};
pub use error::{Error, Result};
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use image_enc::ImageEncoder;
pub use image_view::ImageView;
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use options::GraphOptimizationLevel;
pub use options::{BatchOptions, Options, ThreadOptions};
pub use preproc::{PreprocessedBatch, Preprocessor};
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use siglip2::Siglip2;
#[cfg(feature = "inference")]
#[cfg_attr(docsrs, doc(cfg(feature = "inference")))]
pub use text_enc::TextEncoder;

/// Raw bytes of the bundled `google/siglip2-base-patch16-naflex`
/// `tokenizer.json`, embedded via `include_bytes!`. Used internally
/// by the `bundled` constructors on `TextEncoder` and `Siglip2`;
/// exposed publicly so callers who need to assemble a `Tokenizer`
/// off the bundled JSON (for example, ahead of `from_ort_session`)
/// can do so without round-tripping through disk.
#[cfg(feature = "bundled")]
pub const BUNDLED_TOKENIZER: &[u8] = include_bytes!("../models/tokenizer.json");

/// **Hidden, unstable bench/test helpers — not part of the public API.**
///
/// Exposes the SIMD dispatcher and its scalar fallback by direct name
/// so `benches/bench_dot_768.rs` (and any in-crate microbench) can
/// measure the two paths side-by-side. No semver guarantees on this
/// module — it can change or disappear in any release.
#[doc(hidden)]
pub mod __bench_internal {
  pub fn dot_768_scalar(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::scalar::dot_768(a, b)
  }

  pub fn dot_768_dispatch(a: &[f32], b: &[f32]) -> f32 {
    crate::simd::dot_768(a, b)
  }

  pub fn normalize_patchify_row_scalar(src: &[u8], dst: &mut [f32]) {
    crate::simd::scalar::normalize_patchify_row(src, dst);
  }

  pub fn normalize_patchify_row_dispatch(src: &[u8], dst: &mut [f32]) {
    crate::simd::normalize_patchify_row(src, dst);
  }

  pub fn scale_768_inplace_scalar(v: &mut [f32], factor: f32) {
    crate::simd::scalar::scale_768_inplace(v, factor);
  }

  pub fn scale_768_inplace_dispatch(v: &mut [f32], factor: f32) {
    crate::simd::scale_768_inplace(v, factor);
  }
}
