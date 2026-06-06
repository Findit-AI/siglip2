//! Preprocessing pipeline. For the algorithm for the
//! public `Preprocessor` API.

pub(crate) mod naflex;

use crate::{
  error::{Error, Result},
  image_view::ImageView,
  options::Options,
};

/// Stateless wrapper around the NaFlex preprocessing pipeline. Carries no
/// persistent scratch space in 0.1.0 (`image::imageops::resize` allocates per
/// call, and the patchify/normalize step is small enough that pooling it isn't
/// worth the API cost).
///
/// `Preprocessor: Send + Sync` — guaranteed by the auto-derives because the
/// inner is a `Copy` POD config with no interior mutability. Tests in
/// `tests/integration.rs` carry a compile-time assertion.
#[derive(Clone, Copy, Debug)]
pub struct Preprocessor {
  max_num_patches: u32,
  /// Mirrors `Options::batch.max_batch_size`. Persisted on the
  /// preprocessor so [`Self::new_batch`] can reject capacities above
  /// the cap without re-walking `Options`.
  max_batch_size: usize,
}

impl Preprocessor {
  /// Per-image stride into the `pixel_values` tensor: `MAX_NUM_PATCHES *
  /// 16 * 16 * 3` f32 elements. Buffers passed to
  /// [`Self::preprocess_into`] must match this length exactly.
  pub const BASE_NAFLEX_PIXEL_VALUES_STRIDE: usize = naflex::PIXEL_VALUES_STRIDE;
  /// Per-image stride into the `attention_mask` tensor:
  /// `MAX_NUM_PATCHES` i32 elements (1 = real patch, 0 = pad).
  pub const BASE_NAFLEX_ATTENTION_MASK_STRIDE: usize = naflex::ATTENTION_MASK_STRIDE;
  /// Per-image stride into the `spatial_shapes` tensor: 2 i32 elements
  /// (`[grid_h, grid_w]` for the post-resize patch grid).
  pub const BASE_NAFLEX_SPATIAL_SHAPES_STRIDE: usize = naflex::SPATIAL_SHAPES_STRIDE;

  /// Hard cap baked into the ONNX export. 0.1.0 supports only `max_num_patches = 256`;
  /// passing any other value via `Options` returns `Error::MaxNumPatchesMismatch`.
  pub const MAX_NUM_PATCHES: u32 = 256;

  /// Construct a preprocessor for a given [`Options`]. Returns
  /// [`Error::MaxNumPatchesMismatch`] if the options' `max_num_patches`
  /// is not exactly [`Self::MAX_NUM_PATCHES`] (the only value the
  /// 0.1.0 ONNX export supports), and propagates batch-options
  /// validation errors from `crate::BatchOptions::validate`.
  pub fn new(opts: Options) -> Result<Self> {
    let opt = opts.batch().max_num_patches();
    if opt != Self::MAX_NUM_PATCHES {
      return Err(crate::Error::MaxNumPatchesMismatch {
        opt,
        export: Self::MAX_NUM_PATCHES,
      });
    }

    // Validate the inference micro-batch invariants in one place — the
    // text path calls the same helper so a `Siglip2` built from the
    // same `Options` cannot accept a `BatchOptions` for one modality
    // that the other rejects. See `BatchOptions::validate` for rationale.
    opts.batch().validate()?;

    Ok(Self {
      max_num_patches: opt,
      max_batch_size: opts.batch().max_batch_size(),
    })
  }

  /// The `max_num_patches` value baked into this preprocessor —
  /// always equal to [`Self::MAX_NUM_PATCHES`] for 0.1.0.
  pub fn max_num_patches(&self) -> u32 {
    self.max_num_patches
  }

  /// Mirror of [`crate::BatchOptions::max_batch_size`] — the cap that
  /// [`Self::new_batch`] enforces on requested capacities.
  pub fn max_batch_size(&self) -> usize {
    self.max_batch_size
  }

  /// Writes preprocessed tensors for one image into the supplied buffers.
  /// Buffer lengths must equal the per-image strides above; otherwise returns
  /// `Error::PreprocBufferLength { which }`.
  pub fn preprocess_into(
    &self,
    view: ImageView<'_>,
    pixel_values_out: &mut [f32],
    attention_mask_out: &mut [i32],
    spatial_shapes_out: &mut [i32],
  ) -> Result<()> {
    naflex::preprocess_into(
      view.rgb(),
      view.width(),
      view.height(),
      self.max_num_patches,
      pixel_values_out,
      attention_mask_out,
      spatial_shapes_out,
    )
  }

  /// Allocate a [`PreprocessedBatch`] with capacity for `capacity` images.
  /// Reuse with [`Self::fill_batch`] across calls to avoid per-call
  /// allocation.
  ///
  /// Returns:
  /// - `Error::InvalidBatchSize` if `capacity` is zero or exceeds the
  ///   preprocessor's `max_batch_size` (which closes the
  ///   "request-petabyte-batch" footgun a public caller could otherwise
  ///   reach with `Preprocessor::new_batch(usize::MAX)`).
  /// - `Error::DimensionsOverflow` if `capacity * STRIDE` would wrap
  ///   `usize` (defense in depth against `max_batch_size` raised past
  ///   the wrap point).
  /// - `Error::AllocationFailed` if the global allocator can't satisfy
  ///   the scratch request — the buffer is acquired via
  ///   `Vec::try_reserve_exact` so an oversized request returns this
  ///   typed error instead of aborting the process. At default
  ///   `max_batch_size = 1024` the largest buffer is `pixel_values`
  ///   at ~768 MiB; callers that raise the cap and run on memory-
  ///   constrained hosts can hit this path.
  pub fn new_batch(&self, capacity: usize) -> Result<PreprocessedBatch> {
    if capacity == 0 || capacity > self.max_batch_size {
      return Err(Error::InvalidBatchSize {
        batch_size: capacity,
        max_batch_size: self.max_batch_size,
      });
    }
    let overflow = || Error::DimensionsOverflow {
      width: u32::try_from(capacity).unwrap_or(u32::MAX),
      height: 0,
    };
    let pv_len = capacity
      .checked_mul(Self::BASE_NAFLEX_PIXEL_VALUES_STRIDE)
      .ok_or_else(overflow)?;
    let am_len = capacity
      .checked_mul(Self::BASE_NAFLEX_ATTENTION_MASK_STRIDE)
      .ok_or_else(overflow)?;
    let ss_len = capacity
      .checked_mul(Self::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE)
      .ok_or_else(overflow)?;
    let pixel_values = try_zeroed_vec_f32("pixel_values", pv_len)?;
    let attention_mask = try_zeroed_vec_i32("attention_mask", am_len)?;
    let spatial_shapes = try_zeroed_vec_i32("spatial_shapes", ss_len)?;
    Ok(PreprocessedBatch {
      pixel_values,
      attention_mask,
      spatial_shapes,
      capacity,
      len: 0,
      max_num_patches: self.max_num_patches,
    })
  }

  /// Preprocess `views` into `batch`, replacing any previous contents.
  ///
  /// Returns `Error::BatchTooLarge` if `views.len() > batch.capacity()`.
  /// Aborts on the first failing input with `Error::Batch { index, source }`.
  /// **In every error path the batch is left empty** (`batch.len() == 0`,
  /// `batch.is_empty() == true`) — a stale `len` from a prior successful
  /// fill cannot survive a failed refill, so callers don't have to
  /// special-case the error path before retrying or moving on.
  pub fn fill_batch(&self, batch: &mut PreprocessedBatch, views: &[ImageView<'_>]) -> Result<()> {
    // Clear up front so every early return below leaves the batch in a
    // known-empty state — the previous code's BatchTooLarge path
    // returned without resetting `len`, leaving stale tensors visible
    // to the next `embed_preprocessed(&batch)` call.
    batch.len = 0;
    if views.len() > batch.capacity {
      return Err(Error::BatchTooLarge {
        got: views.len(),
        max: batch.capacity,
      });
    }
    // No pre-zero of the active region: `naflex::preprocess_into` calls
    // `pixel_values_out.fill(0.0)` on its per-image slice as its first
    // step, and the per-image slices below exactly tile
    // `[0..views.len() * STRIDE]`. Pre-zeroing here was a redundant
    // ~768 MiB write at full batch (1024 × 196_608 × 4 bytes), promptly
    // overwritten by `preprocess_into`. The "fill 8 → fill 3 leaks
    // slots 4..8" rationale didn't hold either — `pixel_values_slice()`
    // returns `&pixel_values[..len * STRIDE]`, so slots
    // `[len..old_len)` are unreachable through any public accessor.
    for (i, view) in views.iter().enumerate() {
      let pv_off = i * Self::BASE_NAFLEX_PIXEL_VALUES_STRIDE;
      let am_off = i * Self::BASE_NAFLEX_ATTENTION_MASK_STRIDE;
      let ss_off = i * Self::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE;
      let pv_slice =
        &mut batch.pixel_values[pv_off..pv_off + Self::BASE_NAFLEX_PIXEL_VALUES_STRIDE];
      let am_slice =
        &mut batch.attention_mask[am_off..am_off + Self::BASE_NAFLEX_ATTENTION_MASK_STRIDE];
      let ss_slice =
        &mut batch.spatial_shapes[ss_off..ss_off + Self::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE];
      naflex::preprocess_into(
        view.rgb(),
        view.width(),
        view.height(),
        self.max_num_patches,
        pv_slice,
        am_slice,
        ss_slice,
      )
      // `batch.len` was zeroed at the top of `fill_batch`, so a partial
      // fill stays invisible until the post-loop assignment below
      // commits the new len in one shot.
      .map_err(|source| Error::Batch {
        index: i,
        source: Box::new(source),
      })?;
    }
    batch.len = views.len();
    Ok(())
  }

  /// Allocating shortcut: `new_batch(views.len()) + fill_batch`. Use
  /// `new_batch` + `fill_batch` directly when you want to reuse scratch
  /// across multiple calls.
  pub fn preprocess_batch(&self, views: &[ImageView<'_>]) -> Result<PreprocessedBatch> {
    if views.is_empty() {
      // Sentinel: zero-capacity batch represents an empty input. The
      // ImageEncoder short-circuits on this.
      return Ok(PreprocessedBatch::empty(self.max_num_patches));
    }
    let mut batch = self.new_batch(views.len())?;
    self.fill_batch(&mut batch, views)?;
    Ok(batch)
  }
}

/// Fallible zeroed-`Vec<f32>` allocation. `Vec::try_reserve_exact` is
/// the only `Vec` API that returns a `Result` on allocator failure
/// `vec![0.0; N]` and `Vec::with_capacity(N)` abort the process via
/// the global allocator handler, which we want to avoid for a
/// caller-sized scratch buffer.
#[cfg_attr(not(tarpaulin), inline(always))]
fn try_zeroed_vec_f32(which: &'static str, len: usize) -> Result<Vec<f32>> {
  let mut v: Vec<f32> = Vec::new();
  v.try_reserve_exact(len)
    .map_err(|e| Error::AllocationFailed {
      which,
      requested_bytes: len.saturating_mul(core::mem::size_of::<f32>()),
      cause: e.to_string(),
    })?;
  v.resize(len, 0.0);
  Ok(v)
}

#[cfg_attr(not(tarpaulin), inline(always))]
fn try_zeroed_vec_i32(which: &'static str, len: usize) -> Result<Vec<i32>> {
  let mut v: Vec<i32> = Vec::new();
  v.try_reserve_exact(len)
    .map_err(|e| Error::AllocationFailed {
      which,
      requested_bytes: len.saturating_mul(core::mem::size_of::<i32>()),
      cause: e.to_string(),
    })?;
  v.resize(len, 0);
  Ok(v)
}

/// Opaque preprocessed batch, the only input shape
/// [`crate::ImageEncoder::embed_preprocessed`] accepts.
///
/// Constructible only via [`Preprocessor`] — this rules out the
/// "pass arbitrary `&[f32]` to `embed_preprocessed`" footgun where
/// half-normalized (`x/255 ∈ [0, 1]`), raw u8-as-f32, or otherwise
/// mis-scaled buffers passed every structural check yet produced
/// semantically wrong embeddings.
///
/// The buffers themselves are crate-internal; callers only see
/// length / capacity / `max_num_patches` accessors. Reuse a
/// [`PreprocessedBatch`] across many encoder calls by holding it
/// alongside the encoder and re-filling via
/// [`Preprocessor::fill_batch`] — that keeps the per-call allocation
/// cost at zero.
#[derive(Debug)]
pub struct PreprocessedBatch {
  pixel_values: Vec<f32>,
  attention_mask: Vec<i32>,
  spatial_shapes: Vec<i32>,
  capacity: usize,
  len: usize,
  max_num_patches: u32,
}

impl PreprocessedBatch {
  /// Number of preprocessed images currently in the batch (≤ capacity).
  pub fn len(&self) -> usize {
    self.len
  }
  /// Maximum number of images this batch can hold.
  pub fn capacity(&self) -> usize {
    self.capacity
  }
  /// `true` when no images have been pushed yet.
  pub fn is_empty(&self) -> bool {
    self.len == 0
  }
  /// `max_num_patches` baked into the batch at construction. The
  /// encoder cross-checks this against its own `Options` to refuse
  /// mixing batches built under different patch budgets.
  pub fn max_num_patches(&self) -> u32 {
    self.max_num_patches
  }
  /// Reset to empty without freeing the underlying buffers, so the
  /// next [`Preprocessor::fill_batch`] doesn't reallocate.
  pub fn clear(&mut self) {
    self.len = 0;
  }

  /// Sentinel zero-capacity batch returned by [`Preprocessor::preprocess_batch`]
  /// for an empty input slice. Holds no buffers; embedding it returns
  /// `Ok(vec![])`.
  fn empty(max_num_patches: u32) -> Self {
    Self {
      pixel_values: Vec::new(),
      attention_mask: Vec::new(),
      spatial_shapes: Vec::new(),
      capacity: 0,
      len: 0,
      max_num_patches,
    }
  }

  /// Read-only view of the active `pixel_values` tensor as a flat
  /// `[len() * 256 * 768]` slice of f32, row-major
  /// (image-major / patch-major / channel-innermost).
  ///
  /// Layout: for image `b ∈ 0..len()`, patch `p ∈ 0..256`, channel
  /// `c ∈ 0..768`, the element is at index
  /// `b * 256 * 768 + p * 768 + c`. Padded patch rows
  /// (`p ≥ H_p * W_p` for image `b`, where `(H_p, W_p)` is given by
  /// `spatial_shapes_slice()[b * 2..]`) are zero per the NaFlex
  /// right-pad contract.
  ///
  /// Available without the `inference` feature so wasm / non-ORT
  /// consumers can ship the batch to a different runtime..
  pub fn pixel_values_slice(&self) -> &[f32] {
    &self.pixel_values[..self.len * Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE]
  }

  /// Read-only view of the active `attention_mask` tensor as a flat
  /// `[len() * 256]` slice of i32. Per image `b`, the leading
  /// `H_p * W_p` slots are `1` and the trailing slots are `0`
  /// `(H_p, W_p)` is the corresponding `spatial_shapes_slice()` row.
  pub fn attention_mask_slice(&self) -> &[i32] {
    &self.attention_mask[..self.len * Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE]
  }

  /// Read-only view of the active `spatial_shapes` tensor as a flat
  /// `[len() * 2]` slice of i32. Each image's row is `[H_p, W_p]`
  /// (height-in-patches first), with `H_p * W_p ≤ max_num_patches`.
  pub fn spatial_shapes_slice(&self) -> &[i32] {
    &self.spatial_shapes[..self.len * Preprocessor::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE]
  }
}

#[cfg(test)]
mod tests {
  use super::*;
  use crate::options::BatchOptions;

  #[test]
  fn preprocessor_routes_to_naflex() {
    let opts = Options::default();
    let pre = Preprocessor::new(opts).unwrap();
    assert_eq!(pre.max_num_patches(), 256);

    let rgb = vec![128u8; 16 * 16 * 3];
    let view = ImageView::new(&rgb, 16, 16).unwrap();
    let mut pv = vec![0f32; Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE];
    let mut am = vec![0i32; Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE];
    let mut ss = vec![0i32; Preprocessor::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE];

    pre
      .preprocess_into(view, &mut pv, &mut am, &mut ss)
      .unwrap();
    assert!(ss[0] >= 1 && ss[1] >= 1);
  }

  #[test]
  fn rejects_non_default_max_num_patches() {
    let opts = Options::default().with_batch(BatchOptions::default().with_max_num_patches(512));
    let err = Preprocessor::new(opts).unwrap_err();
    match err {
      crate::Error::MaxNumPatchesMismatch {
        opt: 512,
        export: 256,
      } => {}
      _ => panic!("expected MaxNumPatchesMismatch, got {err}"),
    }
  }

  #[test]
  fn rejects_zero_batch_size() {
    let opts = Options::default().with_batch(BatchOptions::default().with_batch_size(0));
    let err = Preprocessor::new(opts).unwrap_err();
    match err {
      crate::Error::InvalidBatchSize {
        batch_size: 0,
        max_batch_size,
      } => assert_eq!(max_batch_size, 1024),
      _ => panic!("expected InvalidBatchSize, got {err}"),
    }
  }

  #[test]
  fn rejects_batch_size_above_max() {
    // Default max_batch_size = 1024. batch_size = 2048 > max → reject.
    // Closes the over-allocation footgun where a 1-image batch would
    // allocate ~1.5 GB of scratch under batch_size = 2048.
    let opts = Options::default().with_batch(BatchOptions::default().with_batch_size(2048));
    let err = Preprocessor::new(opts).unwrap_err();
    match err {
      crate::Error::InvalidBatchSize {
        batch_size: 2048,
        max_batch_size: 1024,
      } => {}
      _ => panic!("expected InvalidBatchSize, got {err}"),
    }
  }

  #[test]
  fn preprocessor_is_send_sync() {
    fn _req<T: Send + Sync>() {}
    _req::<Preprocessor>();
  }

  /// `Preprocessor::new_batch` allocates buffers sized to the
  /// requested capacity and starts empty.
  #[test]
  fn new_batch_allocates_capacity_and_starts_empty() {
    let pre = Preprocessor::new(Options::default()).unwrap();
    let batch = pre.new_batch(4).unwrap();
    assert_eq!(batch.capacity(), 4);
    assert_eq!(batch.len(), 0);
    assert!(batch.is_empty());
    assert_eq!(batch.max_num_patches(), 256);
  }

  /// Zero capacity is rejected — empty inputs go through
  /// `preprocess_batch` (which returns the empty sentinel) instead.
  #[test]
  fn new_batch_rejects_zero_capacity() {
    let pre = Preprocessor::new(Options::default()).unwrap();
    let err = pre.new_batch(0).unwrap_err();
    match err {
      crate::Error::InvalidBatchSize { batch_size: 0, .. } => {}
      _ => panic!("expected InvalidBatchSize, got {err}"),
    }
  }

  /// a public caller must not be able
  /// to request a `PreprocessedBatch` larger than the configured
  /// `max_batch_size`. Without this cap, `Preprocessor::new_batch(usize::MAX)`
  /// would either allocate petabytes (if the multiplication didn't
  /// wrap) or wrap to a small product on 64-bit and produce a batch
  /// whose advertised capacity exceeds its buffer length, panicking
  /// in subsequent slicing. The cap closes both footguns.
  #[test]
  fn new_batch_rejects_capacity_above_max() {
    // Default max_batch_size = 1024.
    let pre = Preprocessor::new(Options::default()).unwrap();
    let err = pre.new_batch(2048).unwrap_err();
    match err {
      crate::Error::InvalidBatchSize {
        batch_size: 2048,
        max_batch_size: 1024,
      } => {}
      _ => panic!("expected InvalidBatchSize for over-cap, got {err}"),
    }
  }

  /// Belt-and-suspenders: even at the cap boundary, `usize` overflow
  /// in `capacity * STRIDE` is independently guarded via `checked_mul`.
  /// (Unreachable under the `max_batch_size = 1024` default cap, but
  /// the explicit guard documents the invariant.)
  ///
  /// Uses a low custom `max_batch_size` so the at-cap allocation stays
  /// in the single-MiB range. The default cap of 1024 would allocate
  /// `1024 * 256 * 768 * 4 ≈ 768 MiB` of zero-filled f32 for
  /// `pixel_values` alone — fine on a developer machine, OOM on
  /// memory-tight CI runners and a needless slowdown under miri's
  /// instrumented allocator. The cap *value* doesn't matter for what
  /// this test asserts; only the boundary condition does..
  #[test]
  fn new_batch_at_max_cap_succeeds() {
    // 4 patches × 256 × 768 f32 ≈ 3 MiB, the same `capacity ==
    // max_batch_size` boundary case the previous 1024 cap tested.
    // `batch_size` lowered alongside `max_batch_size` because the
    // shared `BatchOptions::validate` rejects `batch_size >
    // max_batch_size`.
    let opts = Options::default().with_batch(
      BatchOptions::default()
        .with_batch_size(4)
        .with_max_batch_size(4),
    );
    let pre = Preprocessor::new(opts).unwrap();
    let batch = pre.new_batch(4).expect("at-cap capacity must succeed");
    assert_eq!(batch.capacity(), 4);
    assert_eq!(batch.len(), 0);
  }

  /// `fill_batch` rejects inputs larger than the batch's capacity
  /// rather than reallocating.
  #[test]
  fn fill_batch_rejects_oversized_input() {
    let pre = Preprocessor::new(Options::default()).unwrap();
    let mut batch = pre.new_batch(2).unwrap();
    let rgb = vec![128u8; 16 * 16 * 3];
    let view = ImageView::new(&rgb, 16, 16).unwrap();
    let views = [view, view, view]; // 3 > capacity 2
    let err = pre.fill_batch(&mut batch, &views).unwrap_err();
    match err {
      crate::Error::BatchTooLarge { got: 3, max: 2 } => {}
      _ => panic!("expected BatchTooLarge, got {err}"),
    }
    // Failure leaves the batch in a known-empty state.
    assert_eq!(batch.len(), 0);
  }

  /// an oversized refill must clear
  /// the batch's `len` so a stale fill from a prior successful call
  /// can't be exposed via `embed_preprocessed(&batch)`. The previous
  /// code returned `BatchTooLarge` before resetting `len`.
  #[test]
  fn oversized_refill_clears_stale_len() {
    let pre = Preprocessor::new(Options::default()).unwrap();
    let mut batch = pre.new_batch(2).unwrap();
    let rgb = vec![128u8; 16 * 16 * 3];
    let view = ImageView::new(&rgb, 16, 16).unwrap();

    // First fill succeeds: batch.len() == 1.
    pre.fill_batch(&mut batch, &[view]).expect("first fill");
    assert_eq!(batch.len(), 1, "first fill should leave len=1");

    // Oversized refill: 3 > capacity 2 → BatchTooLarge.
    let oversized = [view, view, view];
    let err = pre.fill_batch(&mut batch, &oversized).unwrap_err();
    match err {
      crate::Error::BatchTooLarge { .. } => {}
      _ => panic!("expected BatchTooLarge, got {err}"),
    }

    // Stale `len` from the first fill MUST be cleared — without the
    // an earlier version asserted 1 (stale) and a downstream
    // `embed_preprocessed(&batch)` would have re-embedded the prior
    // image instead of failing closed.
    assert_eq!(batch.len(), 0, "oversized refill must clear stale len");
    assert!(batch.is_empty(), "batch must be empty after failed refill");
  }

  /// `fill_batch` followed by `clear` then a smaller fill must not
  /// leak data from the larger fill into the unused slots. Uses the
  /// public `pixel_values_slice` accessor (ungated from the
  /// `inference` feature so wasm/non-ORT consumers can read tensors).
  #[test]
  fn fill_batch_then_smaller_fill_does_not_leak() {
    let pre = Preprocessor::new(Options::default()).unwrap();
    let mut batch = pre.new_batch(4).unwrap();

    let rgb = vec![200u8; 16 * 16 * 3];
    let view = ImageView::new(&rgb, 16, 16).unwrap();
    pre
      .fill_batch(&mut batch, &[view, view, view])
      .expect("first fill");
    assert_eq!(batch.len(), 3);

    batch.clear();
    assert_eq!(batch.len(), 0);

    pre.fill_batch(&mut batch, &[view]).expect("second fill");
    assert_eq!(batch.len(), 1);
    // The pixel_values slice now reflects only slot 0; slots 1..3
    // are not visible because `pixel_values_slice()` is bounded by
    // `len * stride`.
    assert_eq!(
      batch.pixel_values_slice().len(),
      Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE
    );
  }

  /// `preprocess_batch` is the allocating shortcut; an empty input
  /// yields an empty sentinel batch that the encoder short-circuits on.
  #[test]
  fn preprocess_batch_empty_input_yields_empty_batch() {
    let pre = Preprocessor::new(Options::default()).unwrap();
    let batch = pre.preprocess_batch(&[]).unwrap();
    assert!(batch.is_empty());
    assert_eq!(batch.len(), 0);
    assert_eq!(batch.capacity(), 0);
  }

  /// A batch carries the `max_num_patches` it was built under so the
  /// encoder can refuse mixing batches built for different patch
  /// budgets (the only legal value in 0.1.0 is 256, so this is
  /// belt-and-suspenders).
  #[test]
  fn batch_pins_max_num_patches() {
    let pre = Preprocessor::new(Options::default()).unwrap();
    let batch = pre.new_batch(1).unwrap();
    assert_eq!(batch.max_num_patches(), pre.max_num_patches());
  }

  /// oversized scratch requests must
  /// surface as `Error::AllocationFailed` rather than aborting the
  /// process via the global allocator's OOM handler.
  ///
  /// Construction strategy: pick a `max_batch_size` whose
  /// `pixel_values` byte size (`cap * 256 * 768 * 4`) clearly cannot
  /// be allocated by `try_reserve_exact` on any host
  /// (`isize::MAX / 2` bytes, well above 64-bit Rust's effective
  /// max-Vec-bytes ceiling). The `checked_mul` guard catches even
  /// larger values, so we land in a sweet spot: arithmetic succeeds,
  /// allocator refuses, fallible Vec returns the typed error.
  #[test]
  fn new_batch_returns_error_on_alloc_failure() {
    use crate::options::BatchOptions;

    // Pick the smallest `max_batch_size` whose pv bytes overflow what
    // `try_reserve_exact` will ever satisfy. Vec's max byte size is
    // bounded by `isize::MAX`, so requesting more than that returns
    // `TryReserveError`. `(isize::MAX as usize) / (256 * 768 * 4)`
    // ≈ 4.7e12 on 64-bit — but we just need anything >= that. Round
    // up generously.
    let huge = (isize::MAX as usize) / (256 * 768) + 1;

    let opts = Options::default().with_batch(
      BatchOptions::default()
        .with_batch_size(1)
        .with_max_batch_size(huge),
    );
    let pre = Preprocessor::new(opts).expect("preprocessor must accept the options");

    let err = pre.new_batch(huge).unwrap_err();
    match err {
      crate::Error::AllocationFailed {
        which: "pixel_values",
        ..
      } => {}
      _ => panic!("expected AllocationFailed for pixel_values, got {err}"),
    }
  }

  /// the public `*_slice()`
  /// accessors must be reachable without the `inference` feature
  /// so wasm / non-ORT consumers can hand the preprocessed tensors
  /// to a different runtime. This test exercises all three accessors
  /// from the same code path that a no-default-features consumer
  /// would use, asserting shapes, the documented layout, and the
  /// padding contract (mask zero / pixel zero past the active
  /// `H_p * W_p` patches).
  #[test]
  fn batch_accessors_expose_full_tensor_shapes() {
    let pre = Preprocessor::new(Options::default()).unwrap();
    let mut batch = pre.new_batch(2).unwrap();

    // Two distinct synthetic images so the spatial shapes differ
    // from each other.
    let rgb_a = vec![64u8; 16 * 16 * 3];
    let rgb_b = vec![192u8; 32 * 16 * 3];
    let view_a = ImageView::new(&rgb_a, 16, 16).unwrap();
    let view_b = ImageView::new(&rgb_b, 32, 16).unwrap();
    pre.fill_batch(&mut batch, &[view_a, view_b]).expect("fill");
    assert_eq!(batch.len(), 2);

    let pv = batch.pixel_values_slice();
    let am = batch.attention_mask_slice();
    let ss = batch.spatial_shapes_slice();

    // Shapes per the public docstrings.
    assert_eq!(pv.len(), 2 * 256 * 768);
    assert_eq!(am.len(), 2 * 256);
    assert_eq!(ss.len(), 2 * 2);

    // Each spatial-shapes row is [H_p, W_p], both ≥ 1, product ≤ 256.
    for b in 0..2 {
      let h_p = ss[b * 2];
      let w_p = ss[b * 2 + 1];
      assert!(h_p >= 1 && w_p >= 1, "image {b}: H_p={h_p}, W_p={w_p}");
      let n_patches = (h_p as usize) * (w_p as usize);
      assert!(n_patches <= 256, "image {b}: n_patches {n_patches} > 256");

      // Mask layout: leading `n_patches` ones, trailing zeros.
      let am_row = &am[b * 256..(b + 1) * 256];
      for (i, &v) in am_row.iter().enumerate() {
        let want = if i < n_patches { 1 } else { 0 };
        assert_eq!(v, want, "image {b} slot {i}: mask {v}, expected {want}");
      }

      // Padded pixel rows past `n_patches * 768` must be zero.
      let pv_row = &pv[b * 256 * 768..(b + 1) * 256 * 768];
      #[allow(clippy::needless_range_loop)]
      for offset in (n_patches * 768)..(256 * 768) {
        assert_eq!(
          pv_row[offset], 0.0,
          "image {b} padded offset {offset}: non-zero {}",
          pv_row[offset]
        );
      }
    }
  }
}
