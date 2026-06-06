//! MLX (mlxrs) inference backend — Apple-Silicon only.
//!
//! This is the macOS/arm64 alternative to the default `ort` (ONNX Runtime)
//! inference path. It is compiled unconditionally on `aarch64-apple-darwin`
//! (and nowhere else), because `mlxrs` binds the Metal-backed MLX C++ runtime
//! through `mlx-c` FFI and has no other target. There is no `mlx` Cargo feature
//! — the backend is selected automatically by platform (see Cargo.toml).
//!
//! # Design
//!
//! [`crate::ImageEncoder`], [`crate::TextEncoder`], and [`crate::Siglip2`] hold
//! an internal backend enum (`Backend::Ort` vs `Backend::Mlx`). The ONNX path
//! is untouched; the MLX path is reached through the platform auto-routing in
//! the public `from_dir` constructors (which probe the checkpoint directory and
//! pick MLX when an MLX checkpoint is present), never a user-facing backend
//! knob. Both backends expose the **same** public API and return the same
//! [`crate::Embedding`] (768-dim, L2-normalized).
//!
//! # Weight source
//!
//! The MLX backend consumes an **MLX-format checkpoint** (a `config.json` + a
//! weight file + a `tokenizer.json`), not the ONNX graphs. The canonical
//! checkpoint is `google/siglip2-base-patch16-naflex` re-exported to MLX weights
//! (e.g. an `mlx-community` mirror). The loader reads `config.json`, then probes
//! the directory for a weight file in priority order — `model.safetensors`
//! (always), then (when the `npz` feature is on) a `*.npz`, then (when the
//! `gguf` feature is on) a `*.gguf` — loads it via the matching `mlxrs` loader,
//! runs the model's `sanitize` key-remap, and builds the dual tower via `mlxrs`.
//!
//! The model dimensions + quantization scheme are always read from
//! `config.json`; the gguf path is a **weight load seam only** — its embedded
//! metadata is NOT mapped to a config, so a gguf checkpoint still requires a
//! `config.json` alongside it.
//!
//! # Preprocessing
//!
//! The MLX path uses `mlxrs`'s own NaFlex preprocessing
//! (`mlxrs::embeddings::siglip2_naflex::processing::preprocess`), which produces
//! the identical `pixel_values (max_num_patches, 3*P^2)` / `pixel_attention_mask`
//! / `spatial_shapes` tensors this crate's [`crate::preproc`] does — the
//! normalize formula (`(x/255 - 0.5)/0.5`), patch-major channel-innermost
//! layout, and zero-padding all match. The one numeric difference is the resize
//! filter: `mlxrs` resamples with a PIL-bit-exact bilinear (the upstream HF
//! processor's filter) whereas the ONNX path's [`crate::preproc`] uses
//! `image`-rs `Triangle`; the two are close but not byte-identical.

use std::{path::Path, rc::Rc};

use mlxrs::embeddings::siglip2_naflex::{
  Siglip2NaflexModel,
  config::Siglip2NaflexConfig,
  processing::{NaflexInputs, preprocess},
  sanitize,
};

use crate::{
  embedding::Embedding,
  error::{Error, Result},
  image_view::ImageView,
};

/// The standard safetensors weight-file name inside an MLX checkpoint directory
/// — the always-available baseline format the detector probes first.
const SAFETENSORS_FILE: &str = "model.safetensors";
/// The standard config file name inside an MLX checkpoint directory.
const CONFIG_FILE: &str = "config.json";

/// The per-layer quantization marker `mlxrs` (and mlx-embeddings) use: a
/// quantized `nn.Linear` / `nn.Embedding` stores its packed weight alongside a
/// sibling `<prefix>.scales` tensor. Its presence ANYWHERE in the loaded weight
/// map is the sole signal that the checkpoint is quantized (mlx-embeddings'
/// `get_class_predicate`'s `f"{p}.scales" in weights`), exactly the convention
/// [`Siglip2NaflexModel::from_weights_quantized`] keys its per-layer dense-vs-
/// quantized choice on.
const QUANT_SCALES_SUFFIX: &str = ".scales";

/// Discriminate a dense from a quantized MLX checkpoint by the `mlxrs`
/// convention: a quantized checkpoint carries at least one `<layer>.scales`
/// sibling tensor (see [`QUANT_SCALES_SUFFIX`]); a dense one carries none. This
/// is the same `.scales`-presence signal `mlxrs` resolves per layer inside
/// [`Siglip2NaflexModel::from_weights_quantized`] — checked here over the whole
/// (sanitized) weight map to pick the load entry point.
fn weights_are_quantized(weights: &std::collections::HashMap<String, mlxrs::Array>) -> bool {
  weights.keys().any(|k| k.ends_with(QUANT_SCALES_SUFFIX))
}

/// Probe `dir` for an MLX weight file in priority order and load it via the
/// matching `mlxrs` loader, returning the raw (pre-`sanitize`) weight map.
///
/// Priority:
/// 1. `model.safetensors` → [`mlxrs::io::load_safetensors`] (always available).
/// 2. *(feature `npz`)* a `*.npz` (`model.npz` / `weights.npz`, else the single
///    `.npz` in the dir) → [`mlxrs::io::load_npz`].
/// 3. *(feature `gguf`)* a `*.gguf` (`model.gguf`, else the single `.gguf`) →
///    [`mlxrs::io::load_gguf`] (its `.0` weight map; the gguf metadata is NOT
///    mapped to a config — `config.json` is still read separately).
///
/// Returns [`Error::Mlx`] (no MLX checkpoint) when no weight file in any enabled
/// format is present. The npz/gguf branches are compiled only when the
/// corresponding crate feature is enabled, so a default (safetensors-only) build
/// neither compiles nor depends on them.
fn load_weights(dir: &Path) -> Result<std::collections::HashMap<String, mlxrs::Array>> {
  let safetensors = dir.join(SAFETENSORS_FILE);
  if safetensors.is_file() {
    return mlxrs::io::load_safetensors(&safetensors).map_err(Error::from_mlx);
  }

  #[cfg(feature = "npz")]
  if let Some(npz) = find_weight_file(dir, "npz", &["model.npz", "weights.npz"]) {
    return mlxrs::io::load_npz(&npz).map_err(Error::from_mlx);
  }

  #[cfg(feature = "gguf")]
  if let Some(gguf) = find_weight_file(dir, "gguf", &["model.gguf"]) {
    return mlxrs::io::load_gguf(&gguf)
      .map(|(w, _meta)| w)
      .map_err(Error::from_mlx);
  }

  Err(Error::mlx_owned(format!(
    "no MLX checkpoint weight file in {}: expected {SAFETENSORS_FILE}{}",
    dir.display(),
    enabled_format_hint(),
  )))
}

/// Suffix listing the additional weight formats the build accepts, for the
/// "no MLX checkpoint" error. Empty on a default (safetensors-only) build.
fn enabled_format_hint() -> &'static str {
  match (cfg!(feature = "npz"), cfg!(feature = "gguf")) {
    (true, true) => " (or a *.npz / *.gguf)",
    (true, false) => " (or a *.npz)",
    (false, true) => " (or a *.gguf)",
    (false, false) => "",
  }
}

/// Find a single weight file of the given `extension` in `dir`: prefer one of
/// the `preferred` canonical names (in order), else the **sole** file with that
/// extension. Returns `None` if none is present or the choice is ambiguous (more
/// than one candidate and no preferred name matched), so a malformed multi-shard
/// layout falls through to the typed "no checkpoint" error rather than picking
/// an arbitrary file.
///
/// Only referenced from the `npz` / `gguf` detection branches, so it is dead
/// (and `cfg`-elided) on a default build.
#[cfg(any(feature = "npz", feature = "gguf"))]
fn find_weight_file(dir: &Path, extension: &str, preferred: &[&str]) -> Option<std::path::PathBuf> {
  for name in preferred {
    let candidate = dir.join(name);
    if candidate.is_file() {
      return Some(candidate);
    }
  }
  let mut sole: Option<std::path::PathBuf> = None;
  for entry in std::fs::read_dir(dir).ok()?.flatten() {
    let path = entry.path();
    if path.extension().and_then(|e| e.to_str()) == Some(extension) && path.is_file() {
      if sole.is_some() {
        // More than one `.<ext>` file and no preferred name matched: ambiguous.
        return None;
      }
      sole = Some(path);
    }
  }
  sole
}

/// A loaded MLX SigLIP2 NaFlex dual-tower model plus the geometry it needs to
/// preprocess images. Shared (`Rc`) between an [`crate::ImageEncoder`] and a
/// [`crate::TextEncoder`] when both are built from the same checkpoint.
///
/// `mlxrs`'s `Siglip2NaflexModel::encode_*` take `&self`, so this is immutable
/// after construction. `Rc` (not `Arc`) because `Siglip2NaflexModel` is
/// `!Send + !Sync` — it holds MLX device-array handles that are not safe to
/// move or share across threads — so an MLX-backed encoder is single-threaded
/// (`!Send`) by construction, unlike the `ort`-backed `Send` encoder.
#[derive(Clone)]
pub(crate) struct MlxModel {
  model: Rc<Siglip2NaflexModel>,
  /// Vision `patch_size` (px) — the resize/patchify grid cell.
  patch_size: u32,
  /// Vision input channel count (always 3 for the RGB patchify path).
  num_channels: u32,
  /// Per-image patch budget the NaFlex resize targets (`256` for base).
  max_num_patches: u32,
  /// Fixed text sequence length (the text tower's `max_position_embeddings`).
  text_seq_len: usize,
  /// Hard cap on a single batch's length, mirroring the ORT path's
  /// [`crate::BatchOptions::max_batch_size`]. The `from_mlx_dir` constructors
  /// take no [`crate::Options`], so this is the crate-default cap
  /// ([`BatchOptions::default`](crate::BatchOptions)'s `1024`); the image and
  /// text batch paths reject oversized batches with [`Error::BatchTooLarge`]
  /// BEFORE allocating, exactly as the ORT path does.
  max_batch_size: usize,
  /// Micro-batch chunk size for the text path, mirroring the ORT path's
  /// [`crate::BatchOptions::batch_size`]. `embed_text_batch` splits a request
  /// into chunks of this many rows and runs one `encode_text` forward per
  /// chunk, so a within-cap batch never materializes as a single oversized
  /// MLX/Metal graph. The constructors take no [`crate::Options`], so this is
  /// the crate-default (`8`).
  batch_size: usize,
}

impl MlxModel {
  /// Load a model from an MLX checkpoint directory containing `config.json`
  /// and a weight file in any enabled format (`model.safetensors`, or — with
  /// the `npz`/`gguf` features — a `*.npz`/`*.gguf`; see [`load_weights`]).
  pub(crate) fn from_dir(dir: &Path) -> Result<Self> {
    let config_path = dir.join(CONFIG_FILE);

    let config_json = std::fs::read_to_string(&config_path)?;
    let config = Siglip2NaflexConfig::from_json(&config_json).map_err(Error::from_mlx)?;

    // Pull the geometry we need for preprocessing + text padding off the parsed
    // config before it is moved into the model. `try_from` rejects negative /
    // overflowing values, but accepts zero — a zero `patch_size` /
    // `num_channels` / `max_num_patches` / `text_seq_len` reaching the
    // preprocessor would divide-by-zero or build zero-sized tensors, and a
    // non-RGB `num_channels` mismatches the SigLIP2 architecture. So each is
    // additionally validated for positivity (and RGB for the channel count)
    // below. No UPPER cap is imposed on `max_num_patches`: the MLX preprocessor
    // is parameterized by the checkpoint's own declared budget (unlike the ONNX
    // export, which bakes a fixed `256`), so the checkpoint author owns it.
    let patch_size = u32::try_from(config.vision_config.patch_size)
      .ok()
      .filter(|&p| p > 0)
      .ok_or_else(|| Error::mlx("vision patch_size must be a positive dimension"))?;
    let num_channels = u32::try_from(config.vision_config.num_channels)
      .ok()
      .filter(|&c| c == 3)
      .ok_or_else(|| Error::mlx("vision num_channels must be 3 (SigLIP2 is RGB)"))?;
    let max_num_patches = u32::try_from(config.vision_config.max_num_patches())
      .ok()
      .filter(|&n| n > 0)
      .ok_or_else(|| Error::mlx("vision max_num_patches must be a positive dimension"))?;
    let text_seq_len = usize::try_from(config.text_config.max_position_embeddings)
      .ok()
      .filter(|&n| n > 0)
      .ok_or_else(|| Error::mlx("text max_position_embeddings must be a positive length"))?;

    let raw = load_weights(dir)?;
    let weights = sanitize(raw).map_err(Error::from_mlx)?;

    // An MLX SigLIP2 checkpoint may be a QUANTIZED safetensors (mlx-community
    // ships an 8-bit `skip_vision=True` export whose text-tower projections +
    // token embedding carry per-layer `.scales` / `.biases`), not dense f32. Pick
    // the load entry point by the `mlxrs` convention: the presence of any
    // `<layer>.scales` sibling in the (sanitized) weight map is the sole
    // quantized signal. A dense checkpoint (no `.scales`) loads via the plain
    // `from_weights`; a quantized one loads via `from_weights_quantized` with the
    // `(group_size, bits, mode)` parsed from the `config.json` `quantization`
    // block — `mlxrs::lm::quant::parse_quantization` is the golden read-path
    // parser (it returns `None` for a dense config, so it is also the no-op for
    // the dense branch). `from_weights_quantized` then resolves the per-layer
    // dense-vs-quantized split off the same `.scales` markers.
    let model = if weights_are_quantized(&weights) {
      let quantization =
        mlxrs::lm::quant::parse_quantization(&config_json).map_err(Error::from_mlx)?;
      Siglip2NaflexModel::from_weights_quantized(config, weights, quantization.as_ref())
        .map_err(Error::from_mlx)?
    } else {
      Siglip2NaflexModel::from_weights(config, weights).map_err(Error::from_mlx)?
    };

    Ok(Self {
      model: Rc::new(model),
      patch_size,
      num_channels,
      max_num_patches,
      text_seq_len,
      // The MLX constructors take no `Options`, so adopt the crate-default
      // `max_batch_size` cap and `batch_size` micro-batch the ORT path uses by
      // default — same contract.
      max_batch_size: crate::options::BatchOptions::default().max_batch_size(),
      batch_size: crate::options::BatchOptions::default().batch_size(),
    })
  }

  /// The hard per-batch cap this model adopted at construction (the crate-
  /// default `max_batch_size`). Read by [`crate::text_enc::TextEncoder::embed_batch`]
  /// so the outer max-batch guard can reject an oversized batch before the
  /// per-item empty scan, symmetric with the ORT path.
  pub(crate) fn max_batch_size(&self) -> usize {
    self.max_batch_size
  }

  /// NaFlex-preprocess one RGB image into the `mlxrs` device-tensor inputs.
  fn preprocess(&self, view: ImageView<'_>) -> Result<NaflexInputs> {
    preprocess(
      view.rgb(),
      view.width(),
      view.height(),
      self.patch_size,
      self.num_channels,
      self.max_num_patches,
    )
    .map_err(Error::from_mlx)
  }

  /// Encode one decoded RGB image to its 768-dim L2-normalized [`Embedding`].
  pub(crate) fn embed_pixels(&self, view: ImageView<'_>) -> Result<Embedding> {
    let inputs = self.preprocess(view)?;
    let pooled = self.model.encode_image(&inputs).map_err(Error::from_mlx)?;
    // `encode_image` returns `(1, hidden)`; read the single row.
    let rows = eval_rows(&pooled, 1)?;
    embedding_from_row(rows.into_iter().next().expect("1 row requested"))
  }

  /// Encode a batch of decoded RGB images. `mlxrs`'s `encode_image` is
  /// single-image, so this loops; failures are surfaced per-index via
  /// [`Error::Batch`] for parity with the ONNX path.
  pub(crate) fn embed_pixels_batch(&self, views: &[ImageView<'_>]) -> Result<Vec<Embedding>> {
    if views.is_empty() {
      return Ok(Vec::new());
    }
    // Reject oversized batches BEFORE allocating, with the same typed error
    // the ORT path returns (`OrtImageEncoder::embed_pixels_batch`).
    if views.len() > self.max_batch_size {
      return Err(Error::BatchTooLarge {
        got: views.len(),
        max: self.max_batch_size,
      });
    }
    // Fallible reservation (capped at `max_batch_size` above) instead of an
    // infallible `Vec::with_capacity`, so allocation pressure surfaces as a
    // typed error rather than a process abort.
    let mut out = Vec::new();
    out
      .try_reserve_exact(views.len())
      .map_err(|e| Error::AllocationFailed {
        which: "mlx image batch embeddings",
        requested_bytes: views.len().saturating_mul(std::mem::size_of::<Embedding>()),
        cause: e.to_string(),
      })?;
    for (i, view) in views.iter().enumerate() {
      let emb = self.embed_pixels(*view).map_err(|source| Error::Batch {
        index: i,
        source: Box::new(source),
      })?;
      out.push(emb);
    }
    Ok(out)
  }

  /// Encode a batch of text strings. Each string is tokenized with special
  /// tokens, then padded/truncated to the fixed `text_seq_len` under SigLIP's
  /// sticky-EOS contract (matching `mlxrs`'s `Padding::FixedLength`), and run
  /// through the text tower in one call.
  pub(crate) fn embed_text_batch(
    &self,
    tokenizer: &tokenizers::Tokenizer,
    texts: &[&str],
  ) -> Result<Vec<Embedding>> {
    if texts.is_empty() {
      return Ok(Vec::new());
    }
    // Reject oversized batches BEFORE allocating, with the same typed error
    // the ORT path returns (`OrtTextEncoder::embed_batch`).
    if texts.len() > self.max_batch_size {
      return Err(Error::BatchTooLarge {
        got: texts.len(),
        max: self.max_batch_size,
      });
    }
    let seq = self.text_seq_len;
    // Honor the crate's `batch_size` micro-batch contract the ORT path uses
    // (`OrtTextEncoder::embed_batch`'s `texts.chunks(batch_size)` loop): split
    // the (already cap-bounded) request into `batch_size`-row chunks and run one
    // `encode_text` forward per chunk, appending each chunk's rows. A within-cap
    // batch therefore never executes as a single oversized MLX/Metal graph.
    let mut out = Vec::with_capacity(texts.len());
    for (chunk_idx, group) in texts.chunks(self.batch_size).enumerate() {
      // Mirror the ORT path's indexed batch-error contract (`Error::Batch
      // { index, source }`): wrap each chunk-level failure with the chunk's base
      // input index, and a row-level embedding conversion with `base + row`, so a
      // caller can quarantine the offending input regardless of backend.
      let base = chunk_idx * self.batch_size;
      let ids_flat = build_text_input_ids(tokenizer, group, seq).map_err(|e| Error::Batch {
        index: base,
        source: Box::new(e),
      })?;
      let input_ids =
        mlxrs::Array::from_slice::<i32>(&ids_flat, &(group.len(), seq)).map_err(|e| {
          Error::Batch {
            index: base,
            source: Box::new(Error::from_mlx(e)),
          }
        })?;
      let pooled = self
        .model
        .encode_text(&input_ids)
        .map_err(|e| Error::Batch {
          index: base,
          source: Box::new(Error::from_mlx(e)),
        })?;
      for (row_idx, row) in eval_rows(&pooled, group.len())
        .map_err(|e| Error::Batch {
          index: base,
          source: Box::new(e),
        })?
        .into_iter()
        .enumerate()
      {
        out.push(embedding_from_row(row).map_err(|e| Error::Batch {
          index: base + row_idx,
          source: Box::new(e),
        })?);
      }
    }
    Ok(out)
  }
}

/// Tokenize `texts` and build the flat `(batch * seq)` row-major `i32`
/// `input_ids` matrix under SigLIP's sticky-EOS fixed-length contract.
///
/// `tokenizer` MUST have its built-in padding/truncation disabled (see
/// [`crate::text_enc::prepare_mlx_tokenizer`]) so `encode_batch` returns only
/// the real ids (+ the post-processor's appended `<eos>`); each row is then
/// padded/truncated to `seq` by [`fill_fixed_length_row`]. Factored out of
/// [`MlxModel::embed_text_batch`] so the construction is unit-testable without
/// the GPU model.
///
/// `texts.len() * seq` is computed with `checked_mul` and the buffer reserved
/// fallibly, so a pathological `seq` or batch surfaces a typed error, never a
/// panic / abort — matching the ORT path's checked-arithmetic discipline. The
/// caller is responsible for the `max_batch_size` cap (it is enforced in
/// `embed_text_batch` before this call).
pub(crate) fn build_text_input_ids(
  tokenizer: &tokenizers::Tokenizer,
  texts: &[&str],
  seq: usize,
) -> Result<Vec<i32>> {
  let encodings = tokenizer
    .encode_batch(texts.to_vec(), true)
    .map_err(|e| Error::mlx_owned(format!("tokenizer encode failed: {e}")))?;
  let total = texts.len().checked_mul(seq).ok_or_else(|| {
    Error::mlx_owned(format!("batch {} * seq {seq} overflows usize", texts.len()))
  })?;
  let mut ids_flat: Vec<i32> = Vec::new();
  ids_flat
    .try_reserve_exact(total)
    .map_err(|e| Error::AllocationFailed {
      which: "mlx text input_ids",
      requested_bytes: total.saturating_mul(std::mem::size_of::<i32>()),
      cause: e.to_string(),
    })?;
  for enc in &encodings {
    fill_fixed_length_row(enc.get_ids(), seq, &mut ids_flat)?;
  }
  Ok(ids_flat)
}

/// The SigLIP pad / sticky-EOS token id (both `1` — the SigLIP2 sentencepiece
/// `<eos>`). `mlxrs` pads the fixed-length text input with this id and forces it
/// into the final position of a truncated row so the sticky-EOS pooler never
/// sees a content token there. This is the HF SigLIP2 processor's convention
/// (the tokenizer's post-processor appends `<eos>` and the processor pads with
/// it), and it deliberately differs from the standalone ONNX text export, which
/// was re-exported to pad with `<pad>` (id `0`).
const TEXT_PAD_EOS_ID: i32 = 1;

/// Append one fixed-length (`length`) id row to `out` under SigLIP's sticky-EOS
/// contract. `ids` must be the **real** ids `encode` produced (the tokenizer's
/// built-in padding/truncation disabled — see
/// [`crate::text_enc::prepare_mlx_tokenizer`]), i.e. content + the
/// post-processor's appended `<eos>`, at the encoding's natural length.
///
/// - **Within length** (`ids.len() <= length`): every real id is kept and the
///   remaining cells are filled with [`TEXT_PAD_EOS_ID`]. The trailing EOS the
///   post-processor already appended is preserved; the pad cells are real,
///   unmasked positions (the text tower has no attention mask).
/// - **TRUE overlength** (`ids.len() > length`, strictly): the head is kept to
///   `length - 1` and [`TEXT_PAD_EOS_ID`] is forced into the final slot (HF
///   truncate-then-append-EOS), so the sticky-EOS pooler's last position is
///   never a content token.
///
/// The `ids.len() == length` boundary is NOT a truncation — the row is kept
/// verbatim (no EOS forcing). Byte-identical to `mlxrs`'s `build_fixed_length`
/// for `Padding::FixedLength { length, pad_token_id: 1, eos_token_id: Some(1) }`
/// (whose `force_eos` predicate is `ids.len() > length`).
///
/// `tokenizers` ids are `u32`; MLX consumes `input_ids` as `i32` (the embedding
/// gather index dtype). Any id above `i32::MAX` would wrap to a negative index
/// under an `as` cast and silently corrupt the gather (or read out of bounds),
/// so each id is converted with a checked `i32::try_from` and an offending id
/// surfaces as a typed [`Error::Mlx`] rather than silent corruption.
fn fill_fixed_length_row(ids: &[u32], length: usize, out: &mut Vec<i32>) -> Result<()> {
  if length == 0 {
    return Ok(());
  }
  if ids.len() > length {
    // TRUE overlength: head-truncate to `length - 1`, forced EOS in last slot.
    for &id in &ids[..length - 1] {
      out.push(checked_id(id)?);
    }
    out.push(TEXT_PAD_EOS_ID);
  } else {
    // Within length (including the exact `== length` boundary): keep all real
    // ids verbatim, then pad the remainder with the pad/EOS id.
    for &id in ids {
      out.push(checked_id(id)?);
    }
    for _ in ids.len()..length {
      out.push(TEXT_PAD_EOS_ID);
    }
  }
  Ok(())
}

/// Convert a `tokenizers` `u32` id to the `i32` MLX consumes, rejecting any id
/// above `i32::MAX` (which an `as` cast would wrap to a negative gather index)
/// with a typed [`Error::Mlx`] naming the offending id.
fn checked_id(id: u32) -> Result<i32> {
  i32::try_from(id).map_err(|_| {
    Error::mlx_owned(format!(
      "tokenizer id {id} exceeds i32::MAX for MLX input_ids"
    ))
  })
}

/// Evaluate a `(rows, dim)` MLX array and split it into `rows` owned `Vec<f32>`
/// rows. The array is cast to f32 (a no-op for an already-f32 embedding; the
/// needed cast for an f16/bf16/quantized-checkpoint embedding) before `eval` so
/// the model's tensors are never mutated.
fn eval_rows(arr: &mlxrs::Array, rows: usize) -> Result<Vec<Vec<f32>>> {
  let shape = arr.shape();
  if shape.len() != 2 {
    return Err(Error::mlx_owned(format!(
      "expected a rank-2 (rows, dim) embedding tensor, got shape {shape:?}"
    )));
  }
  if shape[0] != rows {
    return Err(Error::mlx_owned(format!(
      "expected {rows} embedding rows, got {}",
      shape[0]
    )));
  }
  let dim = shape[1];
  // Cast to f32 before the host copy: a half-precision (f16/bf16) or quantized
  // MLX checkpoint yields an embedding in its activation dtype (the tower
  // preserves it through `l2_normalize`), and `to_vec::<f32>` is dtype-strict.
  // `astype` is a no-op for an already-f32 embedding and produces a NEW array,
  // so the model's tensors are never mutated (the property the prior
  // `try_clone` had).
  let mut owned = arr.astype(mlxrs::Dtype::F32).map_err(Error::from_mlx)?;
  owned.eval().map_err(Error::from_mlx)?;
  let flat = owned.to_vec::<f32>().map_err(Error::from_mlx)?;
  Ok(flat.chunks_exact(dim).map(<[f32]>::to_vec).collect())
}

/// Wrap one already-L2-normalized model-output row into an [`Embedding`].
///
/// `mlxrs`'s `encode_image` / `encode_text` L2-normalize their output, so the
/// row is unit-norm to f32 ULP; [`Embedding::try_from`] validates the dim (768)
/// and the unit-norm invariant (snapping tiny f32 drift to exactly 1.0), the
/// same validated path the ONNX backend's caller-supplied embeddings take.
fn embedding_from_row(row: Vec<f32>) -> Result<Embedding> {
  Embedding::try_from(row)
}

#[cfg(test)]
mod tests {
  use std::collections::HashMap;

  use super::*;

  const SEQ: usize = 64;

  /// A 1-element f32 [`mlxrs::Array`] stand-in for a weight tensor — the
  /// discriminator only inspects the map's KEYS, never a value's data, so the
  /// shape/contents are immaterial.
  fn dummy() -> mlxrs::Array {
    mlxrs::Array::from_slice::<f32>(&[0.0], &(1,)).expect("1-elem array")
  }

  /// A weight map whose key set mirrors a DENSE checkpoint (only `.weight` /
  /// `.bias` siblings, no `.scales`) is classified dense — it routes to
  /// `from_weights`. This pins the dense side of the dense-vs-quantized load
  /// discriminator without needing the GPU model or a real checkpoint.
  #[test]
  fn discriminator_classifies_dense_weight_map() {
    let mut weights: HashMap<String, mlxrs::Array> = HashMap::new();
    weights.insert(
      "encoder.layers.0.self_attn.q_proj.weight".to_string(),
      dummy(),
    );
    weights.insert(
      "encoder.layers.0.self_attn.q_proj.bias".to_string(),
      dummy(),
    );
    weights.insert("embeddings.token_embedding.weight".to_string(), dummy());
    assert!(
      !weights_are_quantized(&weights),
      "a `.weight`/`.bias`-only map must be classified dense"
    );
  }

  /// A weight map carrying a single `<layer>.scales` sibling (the mlx-community
  /// 8-bit `skip_vision=True` export's quantized text projections) is classified
  /// quantized — it routes to `from_weights_quantized`. A `.scales` ANYWHERE in
  /// the map flips the discriminator, even when most layers stay dense.
  #[test]
  fn discriminator_classifies_quantized_weight_map() {
    let mut weights: HashMap<String, mlxrs::Array> = HashMap::new();
    // A dense vision layer (quantized text-only checkpoints keep vision dense)…
    weights.insert(
      "vision.encoder.layers.0.mlp.fc1.weight".to_string(),
      dummy(),
    );
    // …plus a quantized text projection's packed triple (`.weight` + `.scales`
    // + `.biases`). The `.scales` sibling is the signal.
    weights.insert("text.head.weight".to_string(), dummy());
    weights.insert("text.head.scales".to_string(), dummy());
    weights.insert("text.head.biases".to_string(), dummy());
    assert!(
      weights_are_quantized(&weights),
      "a map with any `.scales` sibling must be classified quantized"
    );
  }

  /// A `.biases` sibling WITHOUT a `.scales` sibling does NOT trip the
  /// discriminator: SigLIP's dense `nn.Linear` also has a bias, and only
  /// `.scales` is the quantization marker (a quantized layer always carries
  /// `.scales`; `.biases` alone is a dense bias). Guards against a false-positive
  /// quantized classification of a dense biased layer.
  #[test]
  fn discriminator_biases_without_scales_is_dense() {
    let mut weights: HashMap<String, mlxrs::Array> = HashMap::new();
    weights.insert("text.head.weight".to_string(), dummy());
    weights.insert("text.head.biases".to_string(), dummy());
    assert!(
      !weights_are_quantized(&weights),
      "a `.biases` sibling without `.scales` is a dense bias, not a quant marker"
    );
  }

  /// Write `config_json` into a fresh temp dir (no `model.safetensors`) and run
  /// [`MlxModel::from_dir`]. The geometry guards run on the parsed config BEFORE
  /// the weights are loaded, so a config rejected for a bad dimension surfaces
  /// without a real checkpoint on disk. Returns the construction `Result`.
  fn from_config_json(tag: &str, config_json: &str) -> Result<MlxModel> {
    let dir = std::env::temp_dir().join(format!(
      "siglip2_mlx_cfg_{tag}_{}_{:?}",
      std::process::id(),
      std::thread::current().id()
    ));
    std::fs::create_dir_all(&dir).expect("create temp config dir");
    std::fs::write(dir.join(CONFIG_FILE), config_json).expect("write config.json");
    let result = MlxModel::from_dir(&dir);
    // Best-effort cleanup; the geometry rejection never created any other file.
    let _ = std::fs::remove_dir_all(&dir);
    result
  }

  /// A `config.json` whose `vision_config`/`text_config` carry the given
  /// overrides (everything else defaulted). The fields under test all have
  /// serde defaults, so an override-only object parses cleanly and the bad
  /// value reaches the [`MlxModel::from_dir`] geometry guard.
  fn config_json_with(vision_overrides: &str, text_overrides: &str) -> String {
    format!(r#"{{"vision_config": {{{vision_overrides}}}, "text_config": {{{text_overrides}}}}}"#)
  }

  /// Each non-positive / non-RGB geometry field is rejected at construction with
  /// a typed [`Error::Mlx`], BEFORE any image / tensor is built — a zero
  /// `patch_size` / `max_num_patches` / `text_seq_len` would otherwise
  /// divide-by-zero or build a zero-sized tensor in the preprocessor, and a
  /// non-RGB `num_channels` mismatches the SigLIP2 architecture. NB: no UPPER
  /// cap is asserted — a large positive `max_num_patches` is the checkpoint
  /// author's budget and stays accepted.
  #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
  #[test]
  fn nonpositive_or_nonrgb_geometry_is_rejected() {
    let cases = [
      ("patch_size", config_json_with("\"patch_size\": 0", "")),
      (
        "num_channels_zero",
        config_json_with("\"num_channels\": 0", ""),
      ),
      (
        "num_channels_gray",
        config_json_with("\"num_channels\": 1", ""),
      ),
      (
        "num_channels_rgba",
        config_json_with("\"num_channels\": 4", ""),
      ),
      (
        "max_num_patches",
        config_json_with("\"max_num_patches\": 0", ""),
      ),
      (
        "text_seq_len",
        config_json_with("", "\"max_position_embeddings\": 0"),
      ),
    ];
    for (tag, json) in cases {
      // `MlxModel` is not `Debug` (it holds an `Rc`-backed device model), so
      // use `.err().expect(...)` rather than `expect_err`.
      let err = from_config_json(tag, &json)
        .err()
        .unwrap_or_else(|| panic!("{tag}: a non-positive/non-RGB dim must be rejected"));
      assert!(
        matches!(err, Error::Mlx(_)),
        "{tag}: expected Error::Mlx, got {err}"
      );
    }
  }

  // NOTE: the end-to-end "a quantized MLX checkpoint loads + embeds via the MLX
  // backend" runtime path is exercised by the gated MLX integration tests in
  // `tests/integration.rs` (the `from_dir` → MLX route): point `SIGLIP2_MLX_DIR`
  // at an mlx-community 8-bit `siglip2-naflex` export and they load it through
  // `from_weights_quantized` here, then assert the image/text embeddings. A
  // dedicated quantized FIXTURE is not committed (the dense PyTorch-reference
  // checkpoint is the pinned parity oracle), so the quantized RUNTIME assertion
  // is deferred to running those gated tests against a quantized checkpoint; the
  // dense-vs-quantized discriminator above is unit-tested unconditionally.

  /// Build a single fixed-length row from real ids — exercises the byte-exact
  /// sticky-EOS fixed-length construction. `ids` are the real ids `encode`
  /// returns once the tokenizer's built-in padding/truncation are disabled.
  fn row(ids: &[u32], length: usize) -> Vec<i32> {
    let mut out = Vec::new();
    fill_fixed_length_row(ids, length, &mut out).expect("ids within i32 range");
    out
  }

  /// A SHORT prompt's real ids (content + trailing EOS=1, e.g. `[a, b, 1]`)
  /// must be kept verbatim and every remaining cell padded with the pad/EOS id
  /// (`1`) — NEVER an interior pad id `0`. Were the tokenizer's built-in
  /// `Fixed(64)`-pad-0 left active, the row would arrive already 64-long with
  /// interior `0`s; disabling it + this builder yields the
  /// correct contract.
  #[test]
  fn short_prompt_pads_with_eos_id_not_interior_zero() {
    let real = [10u32, 20, 30, TEXT_PAD_EOS_ID as u32]; // 3 content + EOS
    let r = row(&real, SEQ);
    assert_eq!(r.len(), SEQ, "row must reach the fixed length");
    // Real ids preserved at the head.
    assert_eq!(&r[..4], &[10, 20, 30, TEXT_PAD_EOS_ID]);
    // Every trailing cell is the pad/EOS id (1), and there is NO interior 0.
    assert!(
      r[4..].iter().all(|&v| v == TEXT_PAD_EOS_ID),
      "trailing cells must be pad/EOS id {TEXT_PAD_EOS_ID}, got {:?}",
      &r[4..]
    );
    assert!(
      !r.contains(&0),
      "a short prompt must not contain pad id 0 anywhere: {r:?}"
    );
  }

  /// The exact `ids.len() == length` boundary is NOT a truncation: the row is
  /// kept verbatim with no EOS forcing — byte-identical to `mlxrs`'s
  /// `build_fixed_length`, whose `force_eos` predicate is strictly
  /// `ids.len() > length`, so the final content token at the boundary is
  /// preserved (an EOS forced here would corrupt it).
  #[test]
  fn exact_length_row_is_verbatim_no_eos_forcing() {
    let real: Vec<u32> = (0..SEQ as u32).collect(); // exactly `length` ids
    let r = row(&real, SEQ);
    assert_eq!(r.len(), SEQ);
    let want: Vec<i32> = (0..SEQ as i32).collect();
    assert_eq!(r, want, "an exactly-length row must be kept verbatim");
    // The final slot is the real last content id (63), not a forced EOS.
    assert_eq!(r[SEQ - 1], (SEQ - 1) as i32);
  }

  /// A TRUE overlength prompt (`ids.len() > length`) keeps its head to
  /// `length - 1` and forces the EOS (`1`) into the final pooled slot
  /// (HF truncate-then-append-EOS — the sticky-EOS pooler must never see a
  /// content token last).
  #[test]
  fn overlength_prompt_forces_sticky_eos_in_last_slot() {
    let real: Vec<u32> = (100..100 + (SEQ as u32 + 50)).collect(); // length + 50 ids
    let r = row(&real, SEQ);
    assert_eq!(r.len(), SEQ);
    // Head is the first `length - 1` real ids.
    let want_head: Vec<i32> = (100..100 + (SEQ as i32 - 1)).collect();
    assert_eq!(&r[..SEQ - 1], want_head.as_slice());
    // Final slot is the forced sticky EOS.
    assert_eq!(r[SEQ - 1], TEXT_PAD_EOS_ID);
  }

  /// The `length == 0` edge yields an empty row and never writes an EOS
  /// (matches `build_fixed_length`'s `length == 0` contract).
  #[test]
  fn zero_length_row_is_empty() {
    assert!(row(&[1, 2, 3], 0).is_empty());
  }

  /// A tokenizer id above `i32::MAX` is rejected with a typed [`Error::Mlx`]
  /// naming the offending id, rather than wrapping (via an `as` cast) to a
  /// negative MLX gather index. Exercised on both the within-length and the
  /// overlength branch — the overlength branch's forced-EOS tail is never
  /// reached because the head conversion fails first.
  #[test]
  fn overflowing_id_is_rejected_not_wrapped() {
    let bad = (i32::MAX as u32) + 1; // first id an i32 cannot hold
    // Within-length branch: the row is shorter than `length`.
    let mut out = Vec::new();
    let err =
      fill_fixed_length_row(&[10, bad, 20], SEQ, &mut out).expect_err("overflow id must reject");
    match err {
      Error::Mlx(msg) => assert!(
        msg.contains(&bad.to_string()),
        "error must name the offending id {bad}, got {msg:?}"
      ),
      other => panic!("expected Error::Mlx, got {other}"),
    }
    // Overlength branch (`ids.len() > length`): same rejection.
    let mut over: Vec<u32> = (0..SEQ as u32 + 10).collect();
    over[0] = bad;
    let mut out2 = Vec::new();
    assert!(
      fill_fixed_length_row(&over, SEQ, &mut out2).is_err(),
      "an overflowing id in the truncated head must reject"
    );
  }

  /// The boundary id `i32::MAX` itself is in range and converts cleanly (no
  /// off-by-one in the `try_from` guard).
  #[test]
  fn max_i32_id_is_accepted() {
    let max = i32::MAX as u32;
    let r = row(&[max], SEQ);
    assert_eq!(r[0], i32::MAX, "i32::MAX id must round-trip, not reject");
  }

  /// A quantized / fp16 MLX checkpoint yields an embedding in f16; the strict
  /// `to_vec::<f32>` would fail without the astype cast. Build an f16 `(2, 2)`
  /// array and assert it extracts to the right f32 rows.
  #[test]
  fn eval_rows_casts_half_precision_embedding_to_f32() {
    let dense = mlxrs::Array::from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &(2, 2)).unwrap();
    let half = dense.astype(mlxrs::Dtype::F16).unwrap();
    let rows = eval_rows(&half, 2).unwrap();
    assert_eq!(rows, vec![vec![1.0_f32, 2.0], vec![3.0, 4.0]]);
  }

  /// Create a fresh temp dir for a format-detection test, named for `tag`.
  fn detect_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
      "siglip2_mlx_detect_{tag}_{}_{:?}",
      std::process::id(),
      std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create temp detect dir");
    dir
  }

  /// `load_weights` over a dir with no weight file in any enabled format returns
  /// the typed "no MLX checkpoint" [`Error::Mlx`] (naming `model.safetensors`),
  /// rather than a panic.
  #[test]
  fn load_weights_no_weight_file_is_typed_error() {
    let dir = detect_dir("none");
    // The `Ok` type (`HashMap<_, Array>`) is not `Debug`, so destructure the
    // `Result` directly rather than via `expect_err`.
    let result = load_weights(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    match result {
      Err(Error::Mlx(msg)) => assert!(
        msg.contains("model.safetensors"),
        "the no-checkpoint error must name model.safetensors, got {msg:?}"
      ),
      Err(other) => panic!("expected Error::Mlx for a weight-less dir, got {other}"),
      Ok(_) => panic!("a dir with no weight file must be rejected"),
    }
  }

  /// The npz weight-file detector prefers a canonical `model.npz` and otherwise
  /// accepts the sole `.npz` in the dir, but reports `None` (ambiguous → falls
  /// through to the typed no-checkpoint error) when several `.npz` files exist
  /// and none is a preferred name. Empty marker files suffice — the detector
  /// only inspects names/extensions, never contents.
  #[cfg(feature = "npz")]
  #[test]
  fn find_weight_file_npz_prefers_canonical_then_sole() {
    let sole = detect_dir("npz_sole");
    std::fs::write(sole.join("export.npz"), b"").expect("write export.npz");
    let picked = find_weight_file(&sole, "npz", &["model.npz", "weights.npz"])
      .expect("the sole .npz must be selected");
    assert_eq!(picked, sole.join("export.npz"));
    let _ = std::fs::remove_dir_all(&sole);

    let canon = detect_dir("npz_canon");
    std::fs::write(canon.join("export.npz"), b"").expect("write export.npz");
    std::fs::write(canon.join("model.npz"), b"").expect("write model.npz");
    let picked = find_weight_file(&canon, "npz", &["model.npz", "weights.npz"])
      .expect("model.npz must win over a non-canonical sibling");
    assert_eq!(picked, canon.join("model.npz"));
    let _ = std::fs::remove_dir_all(&canon);

    let ambig = detect_dir("npz_ambig");
    std::fs::write(ambig.join("a.npz"), b"").expect("write a.npz");
    std::fs::write(ambig.join("b.npz"), b"").expect("write b.npz");
    assert!(
      find_weight_file(&ambig, "npz", &["model.npz", "weights.npz"]).is_none(),
      "two non-canonical .npz files must be ambiguous (None)"
    );
    let _ = std::fs::remove_dir_all(&ambig);
  }

  /// The gguf weight-file detector prefers a canonical `model.gguf` and
  /// otherwise accepts the sole `.gguf` in the dir.
  #[cfg(feature = "gguf")]
  #[test]
  fn find_weight_file_gguf_prefers_canonical_then_sole() {
    let sole = detect_dir("gguf_sole");
    std::fs::write(sole.join("q8.gguf"), b"").expect("write q8.gguf");
    let picked =
      find_weight_file(&sole, "gguf", &["model.gguf"]).expect("the sole .gguf must be selected");
    assert_eq!(picked, sole.join("q8.gguf"));
    let _ = std::fs::remove_dir_all(&sole);

    let canon = detect_dir("gguf_canon");
    std::fs::write(canon.join("q8.gguf"), b"").expect("write q8.gguf");
    std::fs::write(canon.join("model.gguf"), b"").expect("write model.gguf");
    let picked = find_weight_file(&canon, "gguf", &["model.gguf"])
      .expect("model.gguf must win over a non-canonical sibling");
    assert_eq!(picked, canon.join("model.gguf"));
    let _ = std::fs::remove_dir_all(&canon);
  }

  /// `model.safetensors` wins over a present `*.npz`/`*.gguf` regardless of which
  /// features are on: the detector probes safetensors first. With an empty
  /// safetensors marker present, `load_weights` reaches the real
  /// `load_safetensors` (which surfaces an `mlxrs` parse error on the empty
  /// file — proving safetensors was the selected branch, not npz/gguf).
  #[test]
  fn load_weights_prefers_safetensors_first() {
    let dir = detect_dir("prefer_st");
    std::fs::write(dir.join(SAFETENSORS_FILE), b"").expect("write empty safetensors");
    #[cfg(feature = "npz")]
    std::fs::write(dir.join("model.npz"), b"").expect("write model.npz");
    #[cfg(feature = "gguf")]
    std::fs::write(dir.join("model.gguf"), b"").expect("write model.gguf");
    // The `Ok` type is not `Debug`, so destructure the `Result` directly.
    let result = load_weights(&dir);
    let _ = std::fs::remove_dir_all(&dir);
    match result {
      Err(Error::Mlx(_)) => {}
      Err(other) => panic!("expected an Error::Mlx from the safetensors loader, got {other}"),
      Ok(_) => panic!("an empty safetensors must surface a load error, not succeed"),
    }
  }
}
