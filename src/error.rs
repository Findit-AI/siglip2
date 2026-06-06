//! Error type for the full enum and its semantics.

#[cfg(any(feature = "inference", feature = "serde"))]
use std::path::PathBuf;
use thiserror::Error;

/// Helper for the `LoadCalibration` variant only; gated on `serde` to
/// avoid a dead-code warning on `--no-default-features` builds where
/// no other variant uses it.
#[cfg(feature = "serde")]
fn display_loc(path: Option<&PathBuf>) -> String {
  match path {
    Some(p) => format!(" at {}", p.display()),
    None => String::new(),
  }
}

/// Crate-level error type. `#[non_exhaustive]` so adding new variants
/// (or new feature-gated ones) is not a breaking change.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum Error {
  /// ORT-backed graph load failure. Gated on the `inference` feature
  /// because `ort::Error` doesn't exist when the feature is off (wasm
  /// builds, etc.).
  #[cfg(feature = "inference")]
  #[error("failed to load ONNX graph at {path}: {source}")]
  LoadGraph {
    /// Path that was being loaded.
    path: PathBuf,
    /// Underlying ORT error.
    source: ort::Error,
  },

  /// Reserved for future explicit-sidecar-loading APIs. ORT auto-discovers
  /// `.onnx.data` sidecars in 0.1.0, so this variant is currently unused.
  #[cfg(feature = "inference")]
  #[error("failed to load external weights for graph at {path}: {source}")]
  LoadWeights {
    /// Path of the parent ONNX graph whose external weights failed to load.
    path: PathBuf,
    /// Underlying ORT error.
    source: ort::Error,
  },

  /// Image decode failure (JPEG/PNG decode error from the `image` crate).
  /// Only emitted by `ImageEncoder::embed_path` (feature = "decoders").
  #[error("image decode failed: {0}")]
  ImageDecode(
    /// Underlying decoder error message.
    String,
  ),

  /// Required ONNX output tensor was not present in the session output map.
  /// Indicates an unexpected re-export or a corrupted graph.
  #[error("required ONNX output `{name}` was missing from session run")]
  MissingOnnxOutput {
    /// Name of the missing ONNX output tensor.
    name: &'static str,
  },

  /// JSON parse failure for `calibration.json`. Gated on `feature = "serde"`
  /// because the `source` field is `serde_json::Error`, which only exists
  /// when `serde_json` is in the dependency graph. The variant is reachable
  /// only via [`crate::Calibration::from_path`] / [`crate::Calibration::from_bytes`],
  /// which are themselves gated on `feature = "serde"`. `Error` is
  /// `#[non_exhaustive]`, so a feature-conditional variant doesn't break
  /// downstream `match` exhaustiveness contracts.
  #[cfg(feature = "serde")]
  #[error("failed to parse calibration.json{}: {source}", display_loc(.path.as_ref()))]
  LoadCalibration {
    /// Path the JSON was loaded from, or `None` if loaded from bytes.
    path: Option<PathBuf>,
    /// Underlying `serde_json` parse error.
    source: serde_json::Error,
  },

  /// Calibration values failed semantic validation (non-finite values
  /// or out-of-range scalars).
  #[error("invalid calibration values: {reason}")]
  InvalidCalibration {
    /// Human-readable reason the calibration values were rejected.
    reason: &'static str,
  },

  /// Tokenizer load failure (the `tokenizers` crate returned an error).
  #[error("tokenizer load failed: {0}")]
  Tokenizer(
    /// Underlying tokenizer error message.
    String,
  ),

  /// Image had a zero width or height.
  #[error("invalid image dimensions: {width}x{height}")]
  InvalidImage {
    /// Image width in pixels.
    width: u32,
    /// Image height in pixels.
    height: u32,
  },

  /// `width * height * 3` (or `width * height`) overflows `usize` on the
  /// host platform. Not realistically reachable on real images, but legal
  /// `u32` inputs near `u32::MAX` can produce wrap-to-small lengths in
  /// release-mode multiplication, which would let a tiny rgb slice pass
  /// the length check. Caught explicitly via `checked_mul`.
  #[error("image dimensions {width}x{height} overflow usize when computing rgb byte count")]
  DimensionsOverflow {
    /// Image width in pixels.
    width: u32,
    /// Image height in pixels.
    height: u32,
  },

  /// Image dimensions are too large for the NaFlex patch budget at any
  /// positive scale. The post-condition check in `patch_grid` returns this
  /// when `h_p * w_p > max_num_patches`, which can happen for legal `u32`
  /// dimensions near `u32::MAX` even though the algorithm itself converges.
  ///
  #[error(
    "image too large for {max_num_patches}-patch budget: {width}x{height} \
     produced patch grid with {grid_patches} patches"
  )]
  ImageTooLarge {
    /// Image width in pixels.
    width: u32,
    /// Image height in pixels.
    height: u32,
    /// `H_p * W_p` from the patch-grid solver.
    grid_patches: u64,
    /// Configured patch budget that was exceeded.
    max_num_patches: u32,
  },

  /// RGB buffer length did not equal `width * height * 3`.
  #[error("rgb buffer length {got} does not match width*height*3 = {expected}")]
  RgbLength {
    /// Length of the supplied buffer.
    got: usize,
    /// Length the buffer should have had.
    expected: usize,
  },

  /// One of the per-image preprocessed buffers had the wrong length.
  #[error("preprocessed buffer `{which}` length {got} does not match expected {expected}")]
  PreprocBufferLength {
    /// Buffer name (`pixel_values`, `attention_mask`, `spatial_shapes`).
    which: &'static str,
    /// Length of the supplied buffer.
    got: usize,
    /// Length the buffer should have had.
    expected: usize,
  },

  /// Caller-supplied `spatial_shapes` row violates the NaFlex contract:
  /// both dims must be ≥ 1 and `H_p * W_p` must be ≤ `max_num_patches`.
  /// Caught in `embed_preprocessed` before tensors hit ORT so a malformed
  /// caller batch produces a clean error instead of a silently wrong
  /// embedding.
  #[error(
    "spatial_shapes[{batch_index}] = ({h_p}, {w_p}): both dims must be > 0 \
     and product must be ≤ {max_num_patches}"
  )]
  InvalidSpatialShapes {
    /// Index of the offending row in the batch.
    batch_index: usize,
    /// Height-in-patches reported by the caller.
    h_p: i32,
    /// Width-in-patches reported by the caller.
    w_p: i32,
    /// Configured patch budget that bounds `H_p * W_p`.
    max_num_patches: u32,
  },

  /// Caller-supplied `attention_mask` violates the NaFlex contract: must
  /// be exactly `H_p * W_p` leading 1s followed by 0s through the rest
  /// of the 256-slot row.
  #[error(
    "attention_mask[{batch_index}] violates contract at slot {position}: \
     expected {expected}, got {got} (must be {h_w_product} leading ones \
     then zeros)"
  )]
  InvalidAttentionMask {
    /// Index of the offending row in the batch.
    batch_index: usize,
    /// Slot within the row where the contract was first violated.
    position: usize,
    /// Value the slot should have held.
    expected: i32,
    /// Value the caller actually supplied.
    got: i32,
    /// `H_p * W_p` for the row — the count of leading ones expected.
    h_w_product: usize,
  },

  /// Caller-supplied `pixel_values` contains NaN or ±Inf. Sending such
  /// values to ORT propagates them through every matmul in the vision
  /// tower and yields a fully-NaN embedding that downstream cosine code
  /// would silently treat as zero overlap.
  #[error("pixel_values[{batch_index}][{offset}] = {value} is not finite")]
  NonFinitePixelValue {
    /// Index of the offending row in the batch.
    batch_index: usize,
    /// Offset into the row where the non-finite value appears.
    offset: usize,
    /// Non-finite value the caller supplied.
    value: f32,
  },

  /// Caller-supplied `pixel_values` has a non-zero entry in a padded
  /// patch row. The NaFlex contract right-pads the buffer with zero
  /// rows after the first `H_p * W_p` patches; reusing scratch from a
  /// previous call without re-zeroing leaves stale in-range data that
  /// the current graph happens to mask out, but a future re-export
  /// might use those slots, producing silent embedding drift.
  #[error(
    "pixel_values[{batch_index}][{offset}] = {value}: padded patch row \
     (slot {patch_slot} ≥ {n_active_patches} active patches) must be \
     zero per the NaFlex right-pad contract"
  )]
  PaddedPixelNotZero {
    /// Index of the offending row in the batch.
    batch_index: usize,
    /// Offset into the row where the padded value appears.
    offset: usize,
    /// Patch index that should have been zero.
    patch_slot: usize,
    /// Number of active (non-padded) patches in this row.
    n_active_patches: usize,
    /// Stale value left in the padded slot.
    value: f32,
  },

  /// Caller-supplied `pixel_values` contains a value outside the SigLIP
  /// normalized range `[-1.0, 1.0]` (with `1e-3` tolerance for f32 rounding).
  /// The most common cause is passing raw u8-as-f32 pixels (`0..255`) or
  /// half-normalized values (`x/255 ∈ [0, 1]`) — both length- and
  /// mask-correct, both produce plausible-looking but semantically wrong
  /// embeddings that silently corrupt a downstream vector index.
  #[error(
    "pixel_values[{batch_index}][{offset}] = {value} outside SigLIP \
     normalized range [-1, 1] (most likely cause: raw u8 pixels passed \
     without `(x/255 - 0.5) / 0.5` normalization)"
  )]
  PixelValueOutOfRange {
    /// Index of the offending row in the batch.
    batch_index: usize,
    /// Offset into the row where the out-of-range value appears.
    offset: usize,
    /// Out-of-range value the caller supplied.
    value: f32,
  },

  /// ONNX session returned a non-rank-2 output where the encoder
  /// expected `[batch, dim]`.
  #[error("unexpected output rank: expected 2, got {rank} with shape {shape:?}")]
  OutputRank {
    /// Rank (number of dimensions) the session actually returned.
    rank: usize,
    /// Full shape returned by the session.
    shape: Vec<i64>,
  },

  /// Session input shape did not match the encoder's expected pattern.
  #[error("session shape mismatch on `{input}`: expected {expected}, got {got:?}")]
  SessionShapeMismatch {
    /// Name of the offending session input.
    input: &'static str,
    /// Human-readable description of the expected shape.
    expected: &'static str,
    /// Actual shape declared by the session.
    got: Vec<i64>,
  },

  /// Embedding length differed from the expected dimension (768).
  #[error("embedding dimension mismatch: expected {expected}, got {got}")]
  EmbeddingDim {
    /// Expected embedding length.
    expected: usize,
    /// Actual embedding length the session returned.
    got: usize,
  },

  /// Embedding L2 norm fell outside `[1 - epsilon, 1 + epsilon]`.
  #[error("embedding is not unit-norm (got ||v||₂ = {norm}, tolerance ε = {epsilon})")]
  NotNormalized {
    /// Measured L2 norm of the offending embedding.
    norm: f32,
    /// Tolerance used for the unit-norm check.
    epsilon: f32,
  },

  /// Text input was empty (`""`).
  #[error("text input is empty")]
  EmptyText,

  /// Configured `max_num_patches` did not match the value baked into
  /// the bundled ONNX export.
  #[error(
    "max_num_patches in Options ({opt}) does not match the value baked into the ONNX export ({export})"
  )]
  MaxNumPatchesMismatch {
    /// Value supplied via [`crate::BatchOptions::max_num_patches`].
    opt: u32,
    /// Value baked into the ONNX export.
    export: u32,
  },

  /// Caller passed a batch larger than the configured `max_batch_size`.
  #[error("batch size {got} exceeds maximum {max}")]
  BatchTooLarge {
    /// Batch length the caller passed.
    got: usize,
    /// Configured `max_batch_size`.
    max: usize,
  },

  /// `BatchOptions::batch_size` was outside the legal range
  /// `1..=max_batch_size` at encoder construction. Caught at
  /// `Preprocessor::new` so a wrong value can't cause silent over-allocation
  /// later in `embed_pixels_batch`.
  #[error("invalid batch_size {batch_size}: must be in 1..={max_batch_size}")]
  InvalidBatchSize {
    /// Configured `batch_size`.
    batch_size: usize,
    /// Configured `max_batch_size`.
    max_batch_size: usize,
  },

  /// Wraps another `Error` with the index of the batch element that
  /// produced it, so per-element failures stay attributable.
  #[error("batch index {index}: {source}")]
  Batch {
    /// Index of the failing element within the batch.
    index: usize,
    /// Underlying error from that element.
    source: Box<Error>,
  },

  /// `Vec::try_reserve_exact` returned an error — the global allocator
  /// could not satisfy a `PreprocessedBatch` scratch request. This
  /// surfaces as a typed error rather than a process abort. The
  /// numbers below help callers tell whether they hit a cap they
  /// chose (`requested_bytes` matches their `max_batch_size`) versus
  /// system memory pressure.
  ///
  /// `cause` is named to avoid thiserror's `source` field convention
  /// `TryReserveError` doesn't implement `std::error::Error` on stable
  /// Rust today, so we capture its `Display` representation as a string.
  #[error(
    "failed to allocate {requested_bytes} bytes for `{which}` scratch \
     buffer: {cause}"
  )]
  AllocationFailed {
    /// Buffer the allocator was asked to reserve.
    which: &'static str,
    /// Number of bytes that were requested.
    requested_bytes: usize,
    /// `Display` representation of the underlying `TryReserveError`.
    cause: String,
  },

  /// ORT runtime error pass-through. Gated on the `inference` feature
  /// because `ort::Error` doesn't exist when the feature is off.
  #[cfg(feature = "inference")]
  #[error(transparent)]
  Ort(#[from] ort::Error),

  /// `std::io::Error` pass-through (file reads on the calibration /
  /// tokenizer paths, decoder I/O, etc.).
  #[error(transparent)]
  Io(#[from] std::io::Error),

  /// Error from the MLX (`mlxrs`) inference backend — checkpoint load,
  /// NaFlex preprocessing, tokenization, or a tower forward pass. Compiled only
  /// on the Apple-Silicon target, the only place the backend exists. The
  /// `mlxrs::Error` is captured as its `Display` string so this crate's public
  /// `Error` does not leak the `mlxrs` type into its API.
  #[cfg(all(feature = "inference", target_os = "macos", target_arch = "aarch64"))]
  #[error("mlx backend error: {0}")]
  Mlx(
    /// Human-readable description of the MLX backend failure.
    String,
  ),

  /// [`crate::ImageEncoder::embed_preprocessed`] was called on an MLX-backed
  /// encoder. The opaque [`crate::PreprocessedBatch`] is the ONNX path's
  /// normalization-contract carrier; the MLX backend builds its own device
  /// tensors from raw RGB inside `mlxrs`, so it has no use for a
  /// pre-built batch. Call [`crate::ImageEncoder::embed_pixels`] /
  /// [`crate::ImageEncoder::embed_pixels_batch`] instead (both backends
  /// support them). Compiled only on the Apple-Silicon target.
  #[cfg(all(feature = "inference", target_os = "macos", target_arch = "aarch64"))]
  #[error(
    "embed_preprocessed is not supported by the MLX backend; use embed_pixels / \
     embed_pixels_batch (the MLX backend preprocesses raw RGB internally)"
  )]
  MlxUnsupportedPreprocessedBatch,
}

#[cfg(all(feature = "inference", target_os = "macos", target_arch = "aarch64"))]
impl Error {
  /// Build an [`Error::Mlx`] from a static reason string.
  pub(crate) fn mlx(reason: &'static str) -> Self {
    Error::Mlx(reason.to_string())
  }

  /// Build an [`Error::Mlx`] from an owned reason string.
  pub(crate) fn mlx_owned(reason: String) -> Self {
    Error::Mlx(reason)
  }

  /// Convert an `mlxrs::Error` into [`Error::Mlx`], capturing its `Display`.
  pub(crate) fn from_mlx(source: mlxrs::Error) -> Self {
    Error::Mlx(source.to_string())
  }
}

/// Crate-local `Result` alias parameterized over [`enum@Error`].
pub type Result<T> = core::result::Result<T, Error>;

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn invalid_image_displays_dimensions() {
    let err = Error::InvalidImage {
      width: 0,
      height: 480,
    };
    assert_eq!(err.to_string(), "invalid image dimensions: 0x480");
  }

  #[test]
  fn rgb_length_displays_both_lengths() {
    let err = Error::RgbLength {
      got: 100,
      expected: 921_600,
    };
    assert_eq!(
      err.to_string(),
      "rgb buffer length 100 does not match width*height*3 = 921600"
    );
  }

  #[cfg(feature = "serde")]
  #[test]
  fn load_calibration_with_path_includes_location() {
    let bad_json = "{not json";
    let serde_err = serde_json::from_str::<serde_json::Value>(bad_json).unwrap_err();
    let err = Error::LoadCalibration {
      path: Some(PathBuf::from("/tmp/calibration.json")),
      source: serde_err,
    };
    assert!(err.to_string().contains("at /tmp/calibration.json"));
  }

  #[cfg(feature = "serde")]
  #[test]
  fn load_calibration_without_path_omits_location() {
    let bad_json = "{not json";
    let serde_err = serde_json::from_str::<serde_json::Value>(bad_json).unwrap_err();
    let err = Error::LoadCalibration {
      path: None,
      source: serde_err,
    };
    let s = err.to_string();
    // No path-location segment (" at <path>") in the prefix before the colon.
    // (serde_json itself may include " at line N column M" in its message,
    // so we only check the prefix that we control.)
    assert!(
      s.starts_with("failed to parse calibration.json:"),
      "expected prefix 'failed to parse calibration.json:' but got: {s:?}"
    );
    // The controlled prefix must not contain " at " — only the part up to the first ':'
    let prefix = s.split(':').next().unwrap_or("");
    assert!(
      !prefix.contains(" at "),
      "path-location segment appeared in prefix when path is None: {s:?}"
    );
  }
}
