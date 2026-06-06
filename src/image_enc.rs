//! Image encoder. `ImageView` lives in
//! [`crate::image_view`] (always compiled, used by the preprocessor on
//! both wasm and native); this module is gated on `feature = "inference"`
//! and provides the ORT-backed `ImageEncoder`.

use std::path::Path;

use crate::{
  embedding::Embedding,
  error::{Error, Result},
  image_view::ImageView,
  options::Options,
  preproc::{PreprocessedBatch, Preprocessor},
};

/// SigLIP2 NaFlex vision-tower inference.
///
/// Holds either an ONNX Runtime session (the `ort` backend) or — on Apple
/// Silicon, when [`Self::from_dir`] auto-routes to it — an `mlxrs` MLX model
/// (the MLX backend), behind one public API. Both return the same 768-dim
/// L2-normalized [`Embedding`].
///
/// Auto-trait reality is platform-conditional:
/// - On every target **except** `aarch64-apple-darwin`: `Send + !Sync`. Only the
///   `ort` backend is compiled; `ort::Session` is `Send` but `!Sync`.
/// - On `aarch64-apple-darwin`: `!Send + !Sync`. The MLX backend variant is
///   compiled in unconditionally, and it holds an `Rc`-backed MLX model whose
///   device handles are not `Send` — so the enum is `!Send` for *every*
///   `ImageEncoder` on this target, including `ort`-backed ones (an enum is
///   `Send` only if all its variants are).
///
/// Either way the type is `!Sync`. Workers wanting parallelism instantiate one
/// `ImageEncoder` per thread; off Apple Silicon they may alternatively share one
/// behind a `Mutex<ImageEncoder>`.
pub struct ImageEncoder {
  backend: ImageBackend,
}

/// The inference backend behind an [`ImageEncoder`]: the ONNX Runtime path, or
/// (Apple Silicon, auto-routed) the MLX path.
enum ImageBackend {
  Ort(OrtImageEncoder),
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  Mlx(crate::mlx::MlxModel),
}

/// The ONNX Runtime image encoder. Owns one `ort::Session` plus the NaFlex
/// preprocessor and reusable scratch.
struct OrtImageEncoder {
  session: ort::session::Session,
  pre: Preprocessor,
  opts: Options,
  /// Lazily-allocated single-image scratch reused by `embed_pixels`.
  /// Without this, every `embed_pixels` call allocates ~768 KB for
  /// the `pixel_values` buffer; reusing scratch is a meaningful win
  /// for high-throughput callers (search loops, classify-against-many,
  /// etc.).
  embed_pixels_scratch: Option<PreprocessedBatch>,
}

impl ImageEncoder {
  /// Load with default `Options` (Level1 graph optimization, batch_size 8,
  /// single-threaded ORT). The `.onnx.data` external-data sidecar must live
  /// in the same directory as `graph` — ORT auto-discovers it by relative
  /// filename.
  ///
  /// **Not available on wasm32.** `ort 2.0.0-rc.12` cfg-gates
  /// `SessionBuilder::commit_from_file` out on wasm32; wasm callers
  /// must construct an `ort::session::Session` via the wasm-specific
  /// async URL/memory APIs and pass it to [`Self::from_ort_session`].
  #[cfg(not(target_arch = "wasm32"))]
  pub fn from_files(graph: &Path) -> Result<Self> {
    Self::from_files_with_options(graph, Options::default())
  }

  /// Same wasm32 caveat as [`Self::from_files`].
  #[cfg(not(target_arch = "wasm32"))]
  pub fn from_files_with_options(graph: &Path, opts: Options) -> Result<Self> {
    let session = crate::session::build_session(graph, opts)?;
    Self::from_ort_session_with_options(session, opts)
  }

  /// Load the vision tower from a **checkpoint directory**, automatically
  /// picking the best backend for the platform — there is no backend knob.
  ///
  /// On Apple Silicon (`aarch64-apple-darwin`) the directory is probed and routes
  /// to the `mlxrs` **MLX** Metal backend when the vision ONNX graph this
  /// constructor needs (`vision_model_naflex_256.onnx`) is absent and an MLX
  /// checkpoint (`config.json` + `model.safetensors`) is present; otherwise the
  /// **ONNX** vision graph is loaded via ONNX Runtime. On every other platform
  /// only the ONNX backend is compiled, so the ONNX graph is loaded
  /// unconditionally.
  ///
  /// **Not available on wasm32** (the ONNX session constructors are gated out —
  /// see [`Self::from_files`]).
  #[cfg(not(target_arch = "wasm32"))]
  pub fn from_dir(dir: &Path) -> Result<Self> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if crate::backend_select::prefer_mlx(dir, &[crate::backend_select::VISION_ONNX]) {
      return Self::from_mlx_dir(dir);
    }
    Self::from_onnx_dir(dir)
  }

  /// ONNX dispatch target for [`Self::from_dir`]: load the vision ONNX graph
  /// (`crate::backend_select`'s `VISION_ONNX`) from `dir`. Crate-internal — the
  /// user's directory entry point is [`Self::from_dir`].
  #[cfg(not(target_arch = "wasm32"))]
  pub(crate) fn from_onnx_dir(dir: &Path) -> Result<Self> {
    Self::from_files(&dir.join(crate::backend_select::VISION_ONNX))
  }

  /// MLX dispatch target for [`Self::from_dir`]: load from an **MLX checkpoint
  /// directory** (`config.json` + `model.safetensors`) using the `mlxrs` Metal
  /// backend. The vision tower runs on the GPU via MLX.
  ///
  /// Crate-internal — the user's directory entry point is [`Self::from_dir`],
  /// which auto-routes here on Apple Silicon when an MLX checkpoint is present.
  /// The checkpoint is an MLX-format SigLIP2 NaFlex export (e.g. an
  /// `mlx-community` mirror of `google/siglip2-base-patch16-naflex`), not the
  /// ONNX graphs the `from_files` constructors load.
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  pub(crate) fn from_mlx_dir(dir: &Path) -> Result<Self> {
    let model = crate::mlx::MlxModel::from_dir(dir)?;
    Ok(Self {
      backend: ImageBackend::Mlx(model),
    })
  }

  /// Build from a pre-loaded MLX model — used by `Siglip2::from_mlx_dir` so a
  /// single shared model backs both the image and text encoders.
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  pub(crate) fn from_mlx_model(model: crate::mlx::MlxModel) -> Self {
    Self {
      backend: ImageBackend::Mlx(model),
    }
  }

  /// Build from a caller-built session. Validates input/output shapes per
  ///.2 against the SigLIP2-base/naflex/256 contract.
  pub fn from_ort_session(session: ort::session::Session) -> Result<Self> {
    Self::from_ort_session_with_options(session, Options::default())
  }

  fn from_ort_session_with_options(session: ort::session::Session, opts: Options) -> Result<Self> {
    validate_image_session(&session, opts.batch().max_num_patches())?;
    let pre = Preprocessor::new(opts)?;
    Ok(Self {
      backend: ImageBackend::Ort(OrtImageEncoder {
        session,
        pre,
        opts,
        embed_pixels_scratch: None,
      }),
    })
  }

  /// Encode a single pre-decoded RGB image and return its 768-dim
  /// L2-normalized [`Embedding`]. Reuses an internal NaFlex
  /// preprocessing scratch buffer across calls. For batches, prefer
  /// [`Self::embed_pixels_batch`] — it amortizes the per-call ORT
  /// overhead.
  pub fn embed_pixels(&mut self, view: ImageView<'_>) -> Result<Embedding> {
    match &mut self.backend {
      ImageBackend::Ort(ort) => ort.embed_pixels(view),
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      ImageBackend::Mlx(mlx) => mlx.embed_pixels(view),
    }
  }

  /// Returns `Ok(vec![])` for an empty input slice (no inference call).
  /// Returns `Error::BatchTooLarge` when `views.len() > opts.batch.max_batch_size`
  /// (ORT backend). The returned `Vec` preserves input order. Aborts on the
  /// first failing input with `Error::Batch { index, source }`.
  pub fn embed_pixels_batch(&mut self, views: &[ImageView<'_>]) -> Result<Vec<Embedding>> {
    match &mut self.backend {
      ImageBackend::Ort(ort) => ort.embed_pixels_batch(views),
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      ImageBackend::Mlx(mlx) => mlx.embed_pixels_batch(views),
    }
  }

  /// Runs inference on a [`PreprocessedBatch`]. **ORT backend only** — the MLX
  /// backend preprocesses internally via `mlxrs` and returns
  /// [`Error::MlxUnsupportedPreprocessedBatch`] here (the typed
  /// [`PreprocessedBatch`] is the ONNX path's normalization-contract carrier;
  /// the MLX path builds its own device tensors). Use [`Self::embed_pixels`] /
  /// [`Self::embed_pixels_batch`] for both backends.
  pub fn embed_preprocessed(&mut self, batch: &PreprocessedBatch) -> Result<Vec<Embedding>> {
    match &mut self.backend {
      ImageBackend::Ort(ort) => ort.embed_preprocessed(batch),
      #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
      ImageBackend::Mlx(_) => Err(Error::MlxUnsupportedPreprocessedBatch),
    }
  }

  /// Decode JPEG/PNG from disk and call `embed_pixels`. Requires feature
  /// `decoders`. Supported formats: JPEG and PNG only. Works with either
  /// backend (decode is backend-independent).
  #[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
  pub fn embed_path(&mut self, path: &Path) -> Result<Embedding> {
    let img = decode_with_orientation(path)?;
    let (w, h) = img.dimensions();
    let buf = img.into_raw();
    let view = ImageView::new(&buf, w, h)?;
    self.embed_pixels(view)
  }

  /// Run a single throwaway inference to amortize first-call cost (ORT graph
  /// compilation; MLX kernel JIT / weight residency).
  pub fn warmup(&mut self) -> Result<()> {
    let rgb = vec![128u8; 64 * 64 * 3];
    let view = ImageView::new(&rgb, 64, 64)?;
    let _ = self.embed_pixels(view)?;
    Ok(())
  }
}

impl OrtImageEncoder {
  /// Encode a single pre-decoded RGB image and return its 768-dim
  /// L2-normalized [`Embedding`]. Reuses an internal NaFlex
  /// preprocessing scratch buffer across calls. For batches, prefer
  /// [`Self::embed_pixels_batch`] — it amortizes the per-call ORT
  /// overhead.
  fn embed_pixels(&mut self, view: ImageView<'_>) -> Result<Embedding> {
    // Take the scratch out (or lazily allocate on first call), use
    // it, then put it back so the next call reuses the same buffers.
    // The take/put-back dance is what lets us hold `&mut batch` and
    // `&mut self` simultaneously through `fill_batch` + `embed_preprocessed`
    // without aliasing.
    let mut batch = match self.embed_pixels_scratch.take() {
      Some(b) => b,
      None => self.pre.new_batch(1)?,
    };
    let result = self
      .pre
      .fill_batch(&mut batch, &[view])
      .and_then(|()| self.embed_preprocessed(&batch))
      .map(|mut out| out.remove(0));
    // Put scratch back regardless of success — `fill_batch` clears
    // `len` to 0 on error, and a successful run
    // leaves `len = 1` which the next call's `fill_batch` will
    // re-zero before refilling. Either way the batch is safe to reuse.
    self.embed_pixels_scratch = Some(batch);
    result
  }

  /// Returns `Ok(vec![])` for an empty input slice (no ORT call).
  /// Returns `Error::BatchTooLarge` when `views.len() > opts.batch.max_batch_size`.
  /// Internally chunks `views` by `BatchOptions::batch_size`; the returned
  /// `Vec` preserves input order. Aborts on first failing input with
  /// `Error::Batch { index, source }`.
  ///
  /// Allocates one [`PreprocessedBatch`] sized to the smaller of
  /// `batch_size` and `views.len()`, then reuses it across chunks via
  /// [`Preprocessor::fill_batch`] — same per-call alloc cost as the
  /// pre-refactor slice-based path.
  fn embed_pixels_batch(&mut self, views: &[ImageView<'_>]) -> Result<Vec<Embedding>> {
    if views.is_empty() {
      return Ok(Vec::new());
    }
    let max = self.opts.batch().max_batch_size();
    if views.len() > max {
      return Err(Error::BatchTooLarge {
        got: views.len(),
        max,
      });
    }
    let chunk = self.opts.batch().batch_size().max(1);
    // Allocate scratch for the largest possible chunk only — `min(chunk,
    // views.len())`. Without this, `batch_size = 1024` (a valid setting up
    // to `max_batch_size`) would allocate ~770 MB of `pixel_values` even
    // for a single-image batch.
    let alloc_chunk = chunk.min(views.len());
    let mut batch = self.pre.new_batch(alloc_chunk)?;
    let mut out = Vec::with_capacity(views.len());
    for (chunk_idx, group) in views.chunks(chunk).enumerate() {
      self
        .pre
        .fill_batch(&mut batch, group)
        // `fill_batch`'s per-image error returns `Error::Batch { index,
        // .. }` indexed within the chunk — re-wrap to surface the
        // global index for parity with the pre-refactor semantics.
        .map_err(|err| match err {
          Error::Batch { index, source } => Error::Batch {
            index: chunk_idx * chunk + index,
            source,
          },
          other => other,
        })?;
      let chunk_emb = self.embed_preprocessed(&batch)?;
      out.extend(chunk_emb);
    }
    Ok(out)
  }

  /// Runs ONNX on a [`PreprocessedBatch`] produced by [`Preprocessor`].
  /// `batch.is_empty()` returns `Ok(vec![])` without invoking ORT.
  ///
  /// Replaces the older slice-based `embed_preprocessed(&[f32], &[i32],
  /// &[i32], usize)` API. The opaque-type signature rules out the
  /// "pass arbitrary `&[f32]` to the high-throughput path" footgun
  /// where half-normalized (`x/255 ∈ [0, 1]`), raw u8-as-f32, or
  /// otherwise mis-scaled buffers passed every structural check yet
  /// produced semantically wrong embeddings — only [`Preprocessor`]
  /// can build a `PreprocessedBatch`, so the SigLIP normalization
  /// contract is enforced by construction.
  ///
  /// Returns `Error::MaxNumPatchesMismatch` if the batch was built
  /// under a different patch budget than this encoder's `Options`.
  fn embed_preprocessed(&mut self, batch: &PreprocessedBatch) -> Result<Vec<Embedding>> {
    if batch.is_empty() {
      return Ok(Vec::new());
    }
    let opt = self.opts.batch().max_num_patches();
    if opt != batch.max_num_patches() {
      return Err(Error::MaxNumPatchesMismatch {
        opt,
        export: batch.max_num_patches(),
      });
    }
    // Enforce the encoder's `max_batch_size` cap on the typed-batch
    // path. Without this, a caller could build a `PreprocessedBatch`
    // with a separately-configured larger-cap `Preprocessor` and
    // smuggle oversized tensors past the encoder's documented
    // resource guard. Symmetric with `embed_pixels_batch`'s check.
    let max = self.opts.batch().max_batch_size();
    if batch.len() > max {
      return Err(Error::BatchTooLarge {
        got: batch.len(),
        max,
      });
    }
    run_image_session(
      &mut self.session,
      batch.pixel_values_slice(),
      batch.attention_mask_slice(),
      batch.spatial_shapes_slice(),
      batch.len(),
    )
  }
}

/// Open a JPEG/PNG and return its RGB pixels with EXIF orientation
/// applied. The orientation is read off the decoder *before* it is
/// consumed by `DynamicImage::from_decoder` because the trait method
/// takes `&mut self` and `from_decoder` takes ownership; pulling the
/// tag first keeps both calls valid against one decoder. Formats that
/// don't carry an orientation tag (PNG, BMP, …) hit the
/// `ImageDecoder::orientation` trait default, which returns
/// `Orientation::NoTransforms`, so this is a no-op for them.
#[cfg(all(feature = "decoders", not(target_arch = "wasm32")))]
fn decode_with_orientation(path: &Path) -> Result<image::RgbImage> {
  use image::{DynamicImage, ImageDecoder, ImageReader};

  let mut decoder = ImageReader::open(path)?
    .with_guessed_format()?
    .into_decoder()
    .map_err(|e| Error::ImageDecode(e.to_string()))?;
  let orientation = decoder
    .orientation()
    .map_err(|e| Error::ImageDecode(e.to_string()))?;
  let mut img =
    DynamicImage::from_decoder(decoder).map_err(|e| Error::ImageDecode(e.to_string()))?;
  img.apply_orientation(orientation);
  Ok(img.into_rgb8())
}

// `build_session` was moved to the shared `crate::session` module so
// the EP-registration cfg ladder lives in one place. Both
// `ImageEncoder` and `TextEncoder` now call into it.

fn validate_image_session(session: &ort::session::Session, max_num_patches: u32) -> Result<()> {
  use ort::value::TensorElementType;

  let m = max_num_patches as i64;
  let inputs = session.inputs();
  let outputs = session.outputs();

  // Each tuple: (name, expected dtype, expected shape with -1 for dynamic)
  // The batch dim (-1 in expected) accepts any value the graph declares,
  // including -1 (dynamic) or a concrete value. Static dims must match
  // exactly.
  check_outlet(
    inputs,
    "pixel_values",
    TensorElementType::Float32,
    &[-1, m, 768],
  )?;
  check_outlet(
    inputs,
    "pixel_attention_mask",
    TensorElementType::Int32,
    &[-1, m],
  )?;
  check_outlet(inputs, "spatial_shapes", TensorElementType::Int32, &[-1, 2])?;
  check_outlet(
    outputs,
    "pooler_output",
    TensorElementType::Float32,
    &[-1, 768],
  )?;
  // Tighten the contract: assert exact input/output counts. Without this,
  // a future re-export that adds a required input (e.g. an explicit
  // `attention_mask`) would surface as a confusing ORT runtime error at
  // first inference instead of a clean SessionShapeMismatch at construction.
  if inputs.len() != 3 {
    return Err(Error::SessionShapeMismatch {
      input: "<input count>",
      expected: "3 inputs (pixel_values, pixel_attention_mask, spatial_shapes)",
      got: vec![inputs.len() as i64],
    });
  }
  if outputs.len() != 1 {
    return Err(Error::SessionShapeMismatch {
      input: "<output count>",
      expected: "1 output (pooler_output)",
      got: vec![outputs.len() as i64],
    });
  }
  Ok(())
}

/// Crate-internal session-shape check used by both
/// `validate_image_session` and `validate_text_session`.
///
/// `expected_shape` semantics: a value of `-1` is a wildcard (matches any
/// dim including the graph's own `-1` dynamic-batch marker). Any other
/// value must match exactly. The graph's declared shape may itself contain
/// `-1` for dynamic axes; in that case we still accept it (the runtime
/// will catch shape mismatches at inference time, but the static-dim
/// portions of the contract are honored).
pub(crate) fn check_outlet(
  outlets: &[ort::value::Outlet],
  name: &'static str,
  expected_dtype: ort::value::TensorElementType,
  expected_shape: &[i64],
) -> Result<()> {
  use ort::value::ValueType;

  let outlet = outlets
    .iter()
    .find(|o| o.name() == name)
    .ok_or(Error::SessionShapeMismatch {
      input: name,
      expected: "outlet present in session",
      got: vec![],
    })?;

  match outlet.dtype() {
    ValueType::Tensor { ty, shape, .. } => {
      if *ty != expected_dtype {
        return Err(Error::SessionShapeMismatch {
          input: name,
          expected: tensor_element_label(expected_dtype),
          got: shape.to_vec(),
        });
      }
      let actual: &[i64] = shape;
      if actual.len() != expected_shape.len() {
        return Err(Error::SessionShapeMismatch {
          input: name,
          expected: "matching tensor rank",
          got: actual.to_vec(),
        });
      }
      for (i, &want) in expected_shape.iter().enumerate() {
        let act = actual[i];
        if want == -1 {
          // Expected dynamic axis (typically batch). The graph MUST
          // declare it dynamic — accepting a concrete static dim here
          // would let a static-batch graph load successfully and then
          // fail at first `embed_*_batch` call with arbitrary chunk
          // sizes.
          if act != -1 {
            return Err(Error::SessionShapeMismatch {
              input: name,
              expected: "dynamic axis (graph declares -1; static-batch \
                         exports incompatible with batch APIs)",
              got: actual.to_vec(),
            });
          }
        } else {
          // Expected concrete dim. Graph may match exactly or declare
          // the axis dynamic (-1) — both work at runtime.
          if act != -1 && act != want {
            return Err(Error::SessionShapeMismatch {
              input: name,
              expected: "matching static dim",
              got: actual.to_vec(),
            });
          }
        }
      }
      Ok(())
    }
    _ => Err(Error::SessionShapeMismatch {
      input: name,
      expected: "tensor",
      got: vec![],
    }),
  }
}

fn tensor_element_label(t: ort::value::TensorElementType) -> &'static str {
  use ort::value::TensorElementType::*;
  match t {
    Float32 => "f32",
    Float64 => "f64",
    Int8 => "i8",
    Int16 => "i16",
    Int32 => "i32",
    Int64 => "i64",
    Uint8 => "u8",
    Uint16 => "u16",
    Uint32 => "u32",
    Uint64 => "u64",
    Bool => "bool",
    String => "string",
    _ => "<other>",
  }
}

/// Crate-internal invariant tests on slice triples — no production
/// consumer since the public API now goes through
/// [`PreprocessedBatch`], but the tests in [`mod tests`] exercise
/// these to pin the invariants a valid preprocessed batch satisfies.
#[cfg(test)]
fn validate_preprocessed_lengths(
  pv: &[f32],
  am: &[i32],
  ss: &[i32],
  batch_size: usize,
) -> Result<()> {
  // `batch_size * STRIDE` can wrap on 64-bit at pathological inputs
  // (e.g. `batch_size = 1<<63` wraps every stride to 0, letting empty
  // slices pass the length check). `embed_preprocessed` already caps
  // batch_size at `max_batch_size`, so this is unreachable in practice,
  // but `checked_mul` keeps the buffer-length check honest at the
  // boundary by surfacing `DimensionsOverflow` instead of silent wrap.
  let overflow = || Error::DimensionsOverflow {
    width: u32::try_from(batch_size).unwrap_or(u32::MAX),
    height: 0,
  };

  let pv_expected = batch_size
    .checked_mul(Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE)
    .ok_or_else(overflow)?;
  if pv.len() != pv_expected {
    return Err(Error::PreprocBufferLength {
      which: "pixel_values",
      got: pv.len(),
      expected: pv_expected,
    });
  }
  let am_expected = batch_size
    .checked_mul(Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE)
    .ok_or_else(overflow)?;
  if am.len() != am_expected {
    return Err(Error::PreprocBufferLength {
      which: "attention_mask",
      got: am.len(),
      expected: am_expected,
    });
  }
  let ss_expected = batch_size
    .checked_mul(Preprocessor::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE)
    .ok_or_else(overflow)?;
  if ss.len() != ss_expected {
    return Err(Error::PreprocBufferLength {
      which: "spatial_shapes",
      got: ss.len(),
      expected: ss_expected,
    });
  }
  Ok(())
}

/// Structural validation of caller-supplied NaFlex tensors. Runs after
/// `validate_preprocessed_lengths`, so each per-image slice is exactly
/// `[STRIDE]` floats / ints. Rejects:
///
/// - `spatial_shapes[b] = (h_p, w_p)` with either dim ≤ 0 or product
///   exceeding `max_num_patches`. The model derives patch geometry from
///   these values; out-of-range entries either crash ORT or yield
///   silently wrong embeddings that pass the post-inference normalize
///   check.
/// - `attention_mask[b]` not equal to `[1; h_p*w_p] || [0; 256 - h_p*w_p]`.
///   The model uses this mask to ignore padded patch slots; non-binary
///   or wrong-count masks let padded slots leak into attention.
/// - any non-finite value in `pixel_values[b]`. NaN/Inf propagates
///   through every matmul and produces a fully-NaN pooler output.
///
/// Length-correct but malformed batches were a footgun on the
/// earlier slice-based `embed_preprocessed` API. The
/// [`PreprocessedBatch`] opaque type now makes those mistakes
/// unrepresentable at the public API; this validator survives as a
/// crate-internal invariant test (only consumed by [`mod tests`]) and
/// documents what a valid preprocessed batch looks like.
#[cfg(test)]
fn validate_preprocessed_content(
  pv: &[f32],
  am: &[i32],
  ss: &[i32],
  batch_size: usize,
  max_num_patches: u32,
) -> Result<()> {
  let max = max_num_patches as i64;
  for b in 0..batch_size {
    let ss_off = b * Preprocessor::BASE_NAFLEX_SPATIAL_SHAPES_STRIDE;
    let h_p = ss[ss_off];
    let w_p = ss[ss_off + 1];
    if h_p <= 0 || w_p <= 0 {
      return Err(Error::InvalidSpatialShapes {
        batch_index: b,
        h_p,
        w_p,
        max_num_patches,
      });
    }
    let n_patches = (h_p as i64) * (w_p as i64);
    if n_patches > max {
      return Err(Error::InvalidSpatialShapes {
        batch_index: b,
        h_p,
        w_p,
        max_num_patches,
      });
    }

    let am_off = b * Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE;
    let am_row = &am[am_off..am_off + Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE];
    let n_patches_usize = n_patches as usize;
    for (i, &v) in am_row.iter().enumerate() {
      let want: i32 = if i < n_patches_usize { 1 } else { 0 };
      if v != want {
        return Err(Error::InvalidAttentionMask {
          batch_index: b,
          position: i,
          expected: want,
          got: v,
          h_w_product: n_patches_usize,
        });
      }
    }

    let pv_off = b * Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE;
    let pv_row = &pv[pv_off..pv_off + Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE];
    // Per-patch byte stride within a single image's pixel_values: 768
    // floats = 16 (P) × 16 (P) × 3 (channels). The first
    // `n_patches_usize` rows are active; everything from there must
    // be zero per the NaFlex right-pad contract.
    const PATCH_STRIDE: usize = 16 * 16 * 3;
    let active_floats = n_patches_usize * PATCH_STRIDE;
    for (i, &v) in pv_row.iter().enumerate() {
      if !v.is_finite() {
        return Err(Error::NonFinitePixelValue {
          batch_index: b,
          offset: i,
          value: v,
        });
      }
      // SigLIP normalization `(x/255 - 0.5) / 0.5` produces values
      // exactly in `[-1, 1]` for u8 input; bilinear/triangle resize
      // stays in [0, 255], so the post-normalize range is unchanged.
      // Allow `1e-3` for f32 rounding noise. Anything outside this is
      // a length-correct but out-of-domain caller mistake (raw u8
      // passed as f32, etc.).
      const RANGE_BOUND: f32 = 1.0 + 1e-3;
      if v.abs() > RANGE_BOUND {
        return Err(Error::PixelValueOutOfRange {
          batch_index: b,
          offset: i,
          value: v,
        });
      }
      // Padded patch rows must be exactly zero. The current text/vision
      // graph masks via `attention_mask`, but the NaFlex contract is
      // stronger ("right-pad with zeros") and a future re-export could
      // route those slots into attention.
      if i >= active_floats && v != 0.0 {
        return Err(Error::PaddedPixelNotZero {
          batch_index: b,
          offset: i,
          patch_slot: i / PATCH_STRIDE,
          n_active_patches: n_patches_usize,
          value: v,
        });
      }
    }
  }
  Ok(())
}

fn run_image_session(
  session: &mut ort::session::Session,
  pixel_values: &[f32],
  attention_mask: &[i32],
  spatial_shapes: &[i32],
  batch_size: usize,
) -> Result<Vec<Embedding>> {
  use ort::value::TensorRef;

  // Mirror textclap/src/audio.rs: use TensorRef::from_array_view to bind
  // slices without copying. Shape tuples use usize arrays matching the
  // from_array_view (TensorArrayData) trait requirements.
  //
  // Shapes: pixel_values [batch, 256, 768], attention_mask [batch, 256],
  //         spatial_shapes [batch, 2].
  let pv_shape = [batch_size, 256usize, 768];
  let am_shape = [batch_size, 256usize];
  let ss_shape = [batch_size, 2usize];

  let pv_tensor = TensorRef::from_array_view((pv_shape, pixel_values)).map_err(Error::Ort)?;
  let am_tensor = TensorRef::from_array_view((am_shape, attention_mask)).map_err(Error::Ort)?;
  let ss_tensor = TensorRef::from_array_view((ss_shape, spatial_shapes)).map_err(Error::Ort)?;

  // Mirror textclap: session.run(ort::inputs![NAME => tensor]) with ?
  let outputs = session
    .run(ort::inputs![
        "pixel_values" => pv_tensor,
        "pixel_attention_mask" => am_tensor,
        "spatial_shapes" => ss_tensor,
    ])
    .map_err(Error::Ort)?;

  // Mirror textclap: outputs[NAME].try_extract_tensor::<f32>()?
  // Use .get() to avoid panic on missing output.
  let pooler = outputs
    .get("pooler_output")
    .ok_or(Error::MissingOnnxOutput {
      name: "pooler_output",
    })?;
  let (shape, data) = pooler.try_extract_tensor::<f32>().map_err(Error::Ort)?;

  // Shape implements Deref<Target=[i64]> so .len() and index work directly.
  if shape.len() != 2 {
    return Err(Error::OutputRank {
      rank: shape.len(),
      shape: shape.to_vec(),
    });
  }
  if shape[0] != batch_size as i64 || shape[1] != 768 {
    return Err(Error::SessionShapeMismatch {
      input: "pooler_output",
      expected: "[batch, 768]",
      got: shape.to_vec(),
    });
  }

  let mut embeddings = Vec::with_capacity(batch_size);
  for i in 0..batch_size {
    embeddings.push(Embedding::from_model_output(&data[i * 768..(i + 1) * 768])?);
  }
  Ok(embeddings)
}

#[cfg(test)]
mod tests {
  use super::*;

  // ImageView tests moved to `src/image_view.rs` alongside the type
  //.

  /// a graph exported with a CONCRETE
  /// static batch dim (e.g. `[1, 256, 768]` instead of `[-1, 256, 768]`)
  /// would pass the previous wildcard-on-expected -1 check, but our
  /// `embed_pixels_batch` call sends arbitrary per-call batch sizes so
  /// the static-batch graph would fail at first inference. `check_outlet`
  /// must reject static-dim where it expects a dynamic axis.
  #[test]
  fn check_outlet_rejects_static_batch_dim() {
    use ort::value::{Outlet, Shape, SymbolicDimensions, TensorElementType, ValueType};

    // Build an Outlet whose shape is `[1, 256, 768]` (static batch) when
    // we expect `[-1, 256, 768]` (dynamic batch).
    let outlet = Outlet::new(
      "pixel_values",
      ValueType::Tensor {
        ty: TensorElementType::Float32,
        shape: Shape::new([1i64, 256, 768]),
        dimension_symbols: SymbolicDimensions::new([
          String::default(),
          String::default(),
          String::default(),
        ]),
      },
    );
    let outlets = [outlet];
    let err = check_outlet(
      &outlets,
      "pixel_values",
      TensorElementType::Float32,
      &[-1, 256, 768],
    )
    .unwrap_err();
    match err {
      Error::SessionShapeMismatch { input, .. } => {
        assert_eq!(input, "pixel_values");
      }
      _ => panic!("expected SessionShapeMismatch, got {err}"),
    }
  }

  /// Sibling test to `check_outlet_rejects_static_batch_dim`: a graph
  /// with `[-1, 256, 768]` (dynamic batch) must continue to pass.
  #[test]
  fn check_outlet_accepts_dynamic_batch_dim() {
    use ort::value::{Outlet, Shape, SymbolicDimensions, TensorElementType, ValueType};

    let outlet = Outlet::new(
      "pixel_values",
      ValueType::Tensor {
        ty: TensorElementType::Float32,
        shape: Shape::new([-1i64, 256, 768]),
        dimension_symbols: SymbolicDimensions::new([
          "batch".to_string(),
          String::default(),
          String::default(),
        ]),
      },
    );
    let outlets = [outlet];
    check_outlet(
      &outlets,
      "pixel_values",
      TensorElementType::Float32,
      &[-1, 256, 768],
    )
    .expect("dynamic-batch outlet must validate");
  }

  /// `validate_preprocessed_lengths`
  /// multiplied caller-controlled `batch_size` by fixed strides without
  /// `checked_mul`. With `batch_size = 1 << 63` on 64-bit, every stride
  /// product wraps to 0, and empty slices would pass the length check
  /// before ORT was handed nonsensical tensor shapes. Now: caught at the
  /// `embed_preprocessed` `max_batch_size` cap (the first line of
  /// defense), and additionally at the `checked_mul` overflow guard
  /// inside `validate_preprocessed_lengths` (the second line).
  #[test]
  fn validate_preprocessed_lengths_rejects_overflow() {
    // Direct-call the helper at the overflow boundary (the public path
    // `embed_preprocessed` would short-circuit at `BatchTooLarge` first;
    // we want to verify the arithmetic check as a defense-in-depth).
    let pv: Vec<f32> = vec![];
    let am: Vec<i32> = vec![];
    let ss: Vec<i32> = vec![];
    let err = validate_preprocessed_lengths(&pv, &am, &ss, 1usize << 63).unwrap_err();
    match err {
      Error::DimensionsOverflow { .. } => {}
      _ => panic!("expected DimensionsOverflow, got {err}"),
    }
  }

  /// `embed_preprocessed` must reject
  /// length-correct batches whose contents violate the NaFlex contract
  /// (negative/oversized spatial_shapes, malformed attention_mask,
  /// non-finite pixel values). All three rejection paths exercised below.
  fn well_formed_buffers(h_p: i32, w_p: i32) -> (Vec<f32>, Vec<i32>, Vec<i32>) {
    let pv = vec![0.0f32; Preprocessor::BASE_NAFLEX_PIXEL_VALUES_STRIDE];
    let mut am = vec![0i32; Preprocessor::BASE_NAFLEX_ATTENTION_MASK_STRIDE];
    // For non-positive shapes the active region is empty (mask all zeros).
    // The content validator will reject the spatial shape itself; we just
    // want a length-correct batch so the validator gets to the spatial-
    // shape check without underflowing here.
    if h_p > 0 && w_p > 0 {
      let n = (h_p as usize).saturating_mul(w_p as usize);
      for i in 0..n.min(am.len()) {
        am[i] = 1;
      }
    }
    let ss = vec![h_p, w_p];
    (pv, am, ss)
  }

  #[test]
  fn validate_preprocessed_content_rejects_zero_spatial_shape() {
    let (pv, am, ss) = well_formed_buffers(0, 12);
    let err = validate_preprocessed_content(&pv, &am, &ss, 1, 256).unwrap_err();
    match err {
      Error::InvalidSpatialShapes {
        batch_index: 0,
        h_p: 0,
        w_p: 12,
        ..
      } => {}
      _ => panic!("expected InvalidSpatialShapes, got {err}"),
    }
  }

  #[test]
  fn validate_preprocessed_content_rejects_negative_spatial_shape() {
    let (pv, am, ss) = well_formed_buffers(8, -1);
    let err = validate_preprocessed_content(&pv, &am, &ss, 1, 256).unwrap_err();
    match err {
      Error::InvalidSpatialShapes {
        batch_index: 0,
        h_p: 8,
        w_p: -1,
        ..
      } => {}
      _ => panic!("expected InvalidSpatialShapes, got {err}"),
    }
  }

  #[test]
  fn validate_preprocessed_content_rejects_over_budget_spatial_shape() {
    // 17 * 17 = 289 > 256
    let (mut pv, mut am, ss) = well_formed_buffers(17, 17);
    // The mask we built is invalid (n=289 > 256 mask length); rebuild it
    // legally so the spatial-shape check is what fails, not the mask.
    am.iter_mut().for_each(|v| *v = 1);
    pv.iter_mut().for_each(|v| *v = 0.0);
    let err = validate_preprocessed_content(&pv, &am, &ss, 1, 256).unwrap_err();
    match err {
      Error::InvalidSpatialShapes {
        batch_index: 0,
        h_p: 17,
        w_p: 17,
        max_num_patches: 256,
      } => {}
      _ => panic!("expected InvalidSpatialShapes (over budget), got {err}"),
    }
  }

  #[test]
  fn validate_preprocessed_content_rejects_attention_mask_with_zero_in_active_region() {
    // h_p=4, w_p=4 → n_patches=16. Make slot 5 a zero (should be 1).
    let (pv, mut am, ss) = well_formed_buffers(4, 4);
    am[5] = 0;
    let err = validate_preprocessed_content(&pv, &am, &ss, 1, 256).unwrap_err();
    match err {
      Error::InvalidAttentionMask {
        batch_index: 0,
        position: 5,
        expected: 1,
        got: 0,
        h_w_product: 16,
      } => {}
      _ => panic!("expected InvalidAttentionMask, got {err}"),
    }
  }

  #[test]
  fn validate_preprocessed_content_rejects_attention_mask_with_one_in_padding() {
    // h_p=4, w_p=4 → n_patches=16. Make slot 200 a one (should be 0).
    let (pv, mut am, ss) = well_formed_buffers(4, 4);
    am[200] = 1;
    let err = validate_preprocessed_content(&pv, &am, &ss, 1, 256).unwrap_err();
    match err {
      Error::InvalidAttentionMask {
        batch_index: 0,
        position: 200,
        expected: 0,
        got: 1,
        h_w_product: 16,
      } => {}
      _ => panic!("expected InvalidAttentionMask, got {err}"),
    }
  }

  #[test]
  fn validate_preprocessed_content_rejects_non_binary_attention_mask() {
    let (pv, mut am, ss) = well_formed_buffers(4, 4);
    am[3] = 7;
    let err = validate_preprocessed_content(&pv, &am, &ss, 1, 256).unwrap_err();
    match err {
      Error::InvalidAttentionMask {
        position: 3,
        expected: 1,
        got: 7,
        ..
      } => {}
      _ => panic!("expected InvalidAttentionMask, got {err}"),
    }
  }

  #[test]
  fn validate_preprocessed_content_rejects_nan_pixel_value() {
    let (mut pv, am, ss) = well_formed_buffers(4, 4);
    pv[42] = f32::NAN;
    let err = validate_preprocessed_content(&pv, &am, &ss, 1, 256).unwrap_err();
    match err {
      Error::NonFinitePixelValue {
        batch_index: 0,
        offset: 42,
        ..
      } => {}
      _ => panic!("expected NonFinitePixelValue, got {err}"),
    }
  }

  #[test]
  fn validate_preprocessed_content_rejects_infinity_pixel_value() {
    let (mut pv, am, ss) = well_formed_buffers(4, 4);
    pv[1000] = f32::INFINITY;
    let err = validate_preprocessed_content(&pv, &am, &ss, 1, 256).unwrap_err();
    match err {
      Error::NonFinitePixelValue {
        batch_index: 0,
        offset: 1000,
        ..
      } => {}
      _ => panic!("expected NonFinitePixelValue, got {err}"),
    }
  }

  #[test]
  fn validate_preprocessed_content_accepts_well_formed_batch() {
    let (pv, am, ss) = well_formed_buffers(8, 12);
    validate_preprocessed_content(&pv, &am, &ss, 1, 256)
      .expect("well-formed batch must pass content validation");
  }

  /// catch the canonical caller bug of
  /// passing raw u8-as-f32 pixel values (range `[0, 255]`) into the
  /// preprocessed batch API. The buffer is length- and mask-correct, so
  /// only a value-range check stops it from poisoning a downstream
  /// vector index.
  #[test]
  fn validate_preprocessed_content_rejects_raw_u8_as_f32() {
    let (mut pv, am, ss) = well_formed_buffers(8, 12);
    // Fill with a typical mid-gray u8-as-f32 value (128.0). This is what
    // a caller that forgot to apply the SigLIP normalize would emit.
    pv.iter_mut().for_each(|v| *v = 128.0);
    let err = validate_preprocessed_content(&pv, &am, &ss, 1, 256).unwrap_err();
    match err {
      Error::PixelValueOutOfRange {
        batch_index: 0,
        offset: 0,
        value: 128.0,
      } => {}
      _ => panic!("expected PixelValueOutOfRange at offset 0, got {err}"),
    }
  }

  /// Boundary-just-inside: the SigLIP normalization is `(x/255 - 0.5) / 0.5`,
  /// exact range `[-1, 1]`. Allow 1e-3 tolerance for f32 rounding so
  /// real preprocessed embeddings near the boundary aren't rejected.
  #[test]
  fn validate_preprocessed_content_accepts_at_normalized_boundary() {
    let (mut pv, am, ss) = well_formed_buffers(4, 4);
    pv[0] = 1.0;
    pv[1] = -1.0;
    pv[2] = 1.0005; // inside +ε tolerance
    pv[3] = -1.0005;
    validate_preprocessed_content(&pv, &am, &ss, 1, 256)
      .expect("values at the normalized boundary must pass");
  }

  // Round-18: the half-normalized `[0, 1]` gap test that previously
  // documented the validator's partial coverage was deleted in the
  // PreprocessedBatch refactor — half-normalized buffers can no longer
  // reach `embed_preprocessed` because [`crate::PreprocessedBatch`]
  // is constructible only via [`crate::Preprocessor`], which always
  // produces correctly-normalized output. The structural validators
  // below survive as crate-internal invariant tests.

  /// a caller reusing scratch buffers
  /// could leave stale in-range data in `pv_row[n_patches*768..]`. The
  /// current graph masks those slots via `attention_mask`, but the
  /// NaFlex contract requires them to be exactly zero. Caught up
  /// front so a future re-export that routes padded slots into
  /// attention can't produce silent embedding drift.
  #[test]
  fn validate_preprocessed_content_rejects_nonzero_padded_row() {
    let (mut pv, am, ss) = well_formed_buffers(4, 4); // n_patches = 16, active floats = 16*768
    // Stale data in the FIRST padded patch row (slot 16, byte offset 16*768=12288).
    pv[12288] = 0.5; // legal range, but in a padded slot
    let err = validate_preprocessed_content(&pv, &am, &ss, 1, 256).unwrap_err();
    match err {
      Error::PaddedPixelNotZero {
        batch_index: 0,
        offset: 12288,
        patch_slot: 16,
        n_active_patches: 16,
        ..
      } => {}
      _ => panic!("expected PaddedPixelNotZero, got {err}"),
    }
  }

  /// Boundary-just-outside: a single out-of-range value past the tolerance
  /// is rejected with the specific offending offset surfaced.
  #[test]
  fn validate_preprocessed_content_rejects_just_outside_range() {
    let (mut pv, am, ss) = well_formed_buffers(4, 4);
    pv[100] = 1.5; // ~50% beyond the legal range — clearly mis-scaled
    let err = validate_preprocessed_content(&pv, &am, &ss, 1, 256).unwrap_err();
    match err {
      Error::PixelValueOutOfRange {
        batch_index: 0,
        offset: 100,
        ..
      } => {}
      _ => panic!("expected PixelValueOutOfRange at offset 100, got {err}"),
    }
  }
}
