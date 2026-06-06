//! `Siglip2` wrapper.

// `Path` and `Options` are used by the `from_files*` (gated on
// `feature = "serde"`) and `bundled*` (gated on `feature = "bundled"`)
// constructors. Gate the imports on the union so neither feature
// alone produces unused-import warnings under `-D warnings`.
#[cfg(any(feature = "serde", feature = "bundled"))]
use std::path::Path;

use tokenizers::Tokenizer;

#[cfg(any(feature = "serde", feature = "bundled"))]
use crate::options::Options;

use crate::{
  calibration::Calibration, embedding::LabeledScore, error::Result, image_enc::ImageEncoder,
  image_view::ImageView, text_enc::TextEncoder,
};

/// High-level wrapper holding both encoders plus calibration.
pub struct Siglip2 {
  image: ImageEncoder,
  text: TextEncoder,
  calibration: Calibration,
}

impl Siglip2 {
  /// **Not available on wasm32.** See
  /// [`crate::image_enc::ImageEncoder::from_files`] for rationale
  /// `ort 2.0.0-rc.12` cfg-gates `commit_from_file` out of wasm32
  /// builds. On wasm callers must construct both `ort::session::Session`
  /// values via the wasm-specific async APIs and pass them to
  /// [`Self::from_parts`].
  #[cfg(all(feature = "serde", not(target_arch = "wasm32")))]
  pub fn from_files(
    vision_onnx: &Path,
    text_onnx: &Path,
    tokenizer_json: &Path,
    calibration_json: &Path,
  ) -> Result<Self> {
    Self::from_files_with_options(
      vision_onnx,
      text_onnx,
      tokenizer_json,
      calibration_json,
      Options::default(),
    )
  }

  /// Same wasm32 caveat as [`Self::from_files`]. Gated on
  /// `feature = "serde"` because `Calibration::from_path` parses the
  /// `calibration.json` via `serde_json`.
  #[cfg(all(feature = "serde", not(target_arch = "wasm32")))]
  pub fn from_files_with_options(
    vision_onnx: &Path,
    text_onnx: &Path,
    tokenizer_json: &Path,
    calibration_json: &Path,
    opts: Options,
  ) -> Result<Self> {
    let image = ImageEncoder::from_files_with_options(vision_onnx, opts)?;
    let text = TextEncoder::from_files_with_options(text_onnx, tokenizer_json, opts)?;
    let calibration = Calibration::from_path(calibration_json)?;
    Ok(Self {
      image,
      text,
      calibration,
    })
  }

  /// Same wasm32 caveat as `Self::from_files` (gated on
  /// `feature = "serde"`, so the link can't be intra-doc here).
  /// Calibration is baked
  /// in at build time from `models/siglip2/calibration.json` (see
  /// [`Calibration::bundled`]), so this constructor doesn't depend on
  /// `feature = "serde"` — `--no-default-features --features
  /// inference,bundled` produces a working `Siglip2` with no serde in
  /// the dependency tree.
  #[cfg(all(feature = "bundled", not(target_arch = "wasm32")))]
  pub fn bundled(vision_onnx: &Path, text_onnx: &Path) -> Result<Self> {
    Self::bundled_with_options(vision_onnx, text_onnx, Options::default())
  }

  /// Same wasm32 caveat as `Self::from_files`. See [`Self::bundled`]
  /// for the rationale on the dropped `calibration_json` parameter and
  /// the absence of a `feature = "serde"` gate.
  #[cfg(all(feature = "bundled", not(target_arch = "wasm32")))]
  pub fn bundled_with_options(vision_onnx: &Path, text_onnx: &Path, opts: Options) -> Result<Self> {
    let image = ImageEncoder::from_files_with_options(vision_onnx, opts)?;
    let text = TextEncoder::bundled_with_options(text_onnx, opts)?;
    let calibration = Calibration::bundled();
    Ok(Self {
      image,
      text,
      calibration,
    })
  }

  /// Load both encoders from a **checkpoint directory**, automatically picking
  /// the best backend for the platform — there is no backend knob.
  ///
  /// On Apple Silicon (`aarch64-apple-darwin`) the directory is probed and routes
  /// to the `mlxrs` **MLX** Metal backend when **neither** of the ONNX graphs
  /// this constructor needs (`vision_model_naflex_256.onnx` and
  /// `text_model_naflex.onnx`) is present and an MLX checkpoint
  /// (`config.json` plus `model.safetensors`) is present; otherwise the **ONNX**
  /// Runtime backend is used. Because `Siglip2` loads both towers, the presence
  /// of *either* ONNX graph routes the whole wrapper to ONNX. On every other
  /// platform only the ONNX backend is compiled, so an ONNX checkpoint is loaded
  /// unconditionally.
  ///
  /// The tokenizer is `tokenizer.json` in the same directory; calibration is
  /// the bundled pinned release values ([`Calibration::bundled`]).
  ///
  /// Requires `feature = "bundled"` (for the pinned [`Calibration`] and, on the
  /// ONNX path, the bundled tokenizer constructors). **Not available on
  /// wasm32** (the ONNX session constructors are gated out there — see
  /// [`Self::bundled`]).
  #[cfg(all(feature = "bundled", not(target_arch = "wasm32")))]
  pub fn from_dir(dir: &Path) -> Result<Self> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    if crate::backend_select::prefer_mlx(
      dir,
      &[
        crate::backend_select::VISION_ONNX,
        crate::backend_select::TEXT_ONNX,
      ],
    ) {
      return Self::from_mlx_dir(dir);
    }
    Self::from_onnx_dir(dir)
  }

  /// ONNX dispatch target for [`Self::from_dir`]: load the SigLIP2 NaFlex ONNX
  /// graphs (`crate::backend_select`'s `VISION_ONNX` / `TEXT_ONNX`) from `dir`,
  /// with the tokenizer read from **`dir/tokenizer.json`** and calibration from
  /// the bundled pinned release values ([`Calibration::bundled`]). Kept
  /// crate-internal — the user's entry point is [`Self::from_dir`] (or the
  /// explicit [`Self::from_files`] / [`Self::bundled`] file-path constructors).
  ///
  /// Composed from the per-encoder `from_onnx_dir` constructors rather than
  /// [`Self::bundled`]: the latter loads the embedded `BUNDLED_TOKENIZER`, which
  /// would silently produce wrong token ids for an ONNX checkpoint / fine-tune
  /// shipping its own `tokenizer.json`. [`TextEncoder::from_onnx_dir`] reads the
  /// directory's tokenizer (mirroring its sibling [`ImageEncoder::from_onnx_dir`]
  /// for the vision graph), so the assembled `Siglip2` honors the checkpoint's
  /// own tokenizer.
  #[cfg(all(feature = "bundled", not(target_arch = "wasm32")))]
  pub(crate) fn from_onnx_dir(dir: &Path) -> Result<Self> {
    let image = ImageEncoder::from_onnx_dir(dir)?;
    let text = TextEncoder::from_onnx_dir(dir)?;
    let calibration = Calibration::bundled();
    Ok(Self {
      image,
      text,
      calibration,
    })
  }

  /// MLX dispatch target for [`Self::from_dir`]: load from an **MLX checkpoint
  /// directory** (`config.json` + `model.safetensors` + `tokenizer.json`) using
  /// the `mlxrs` Metal backend. A single shared model backs both the image and
  /// text encoders.
  ///
  /// Kept crate-internal — the user's entry point is [`Self::from_dir`], which
  /// auto-routes here on Apple Silicon when an MLX checkpoint is present.
  /// Calibration uses the bundled pinned release values
  /// ([`Calibration::bundled`]) — an MLX checkpoint's `config.json` carries no
  /// sigmoid scale/bias, but the model's own `logit_scale` / `logit_bias`
  /// tensors are applied directly by `mlxrs` for raw-logit scoring; the
  /// bundled `Calibration` here is what [`Self::classify`] uses for the
  /// `sigmoid(exp(scale)·cos + bias)` probability.
  #[cfg(all(feature = "bundled", target_os = "macos", target_arch = "aarch64"))]
  pub(crate) fn from_mlx_dir(dir: &Path) -> Result<Self> {
    let model = crate::mlx::MlxModel::from_dir(dir)?;
    // Prepare the tokenizer the same way `TextEncoder::from_mlx_dir` does:
    // disable the serialized padding/truncation and install the SigLIP2
    // lowercasing normalizer (the MLX path builds the fixed-length-64 row
    // manually under the sticky-EOS contract).
    let tokenizer = crate::text_enc::prepare_mlx_tokenizer(&dir.join("tokenizer.json"))?;
    let image = ImageEncoder::from_mlx_model(model.clone());
    let text = TextEncoder::from_mlx_model(model, tokenizer);
    let calibration = Calibration::bundled();
    Ok(Self {
      image,
      text,
      calibration,
    })
  }

  /// Build from caller-owned components. **Re-validates `calibration`** through
  /// the same pipeline as `Calibration::from_path` / `from_bytes` use, so a
  /// hand-built `Calibration::new(NaN, NaN)` cannot reach `classify`.
  pub fn from_parts(
    image_session: ort::session::Session,
    text_session: ort::session::Session,
    tokenizer: Tokenizer,
    calibration: Calibration,
  ) -> Result<Self> {
    let calibration = Calibration::validate(calibration.logit_scale(), calibration.logit_bias())?;
    let image = ImageEncoder::from_ort_session(image_session)?;
    let text = TextEncoder::from_ort_session(text_session, tokenizer)?;
    Ok(Self {
      image,
      text,
      calibration,
    })
  }

  /// Mutable access to the inner [`ImageEncoder`] — useful for
  /// calling `embed_pixels` / `embed_pixels_batch` directly without
  /// going through `classify`.
  pub fn image(&mut self) -> &mut ImageEncoder {
    &mut self.image
  }

  /// Mutable access to the inner [`TextEncoder`] — useful for
  /// calling `embed` / `embed_batch` directly without going through
  /// `classify`.
  pub fn text(&mut self) -> &mut TextEncoder {
    &mut self.text
  }

  /// Borrow both encoders simultaneously. `classify` uses this internally.
  pub fn split(&mut self) -> (&mut ImageEncoder, &mut TextEncoder) {
    (&mut self.image, &mut self.text)
  }

  /// Warm up both encoders. Equivalent to `self.image().warmup(); self.text().warmup();`.
  pub fn warmup(&mut self) -> crate::error::Result<()> {
    let (img, txt) = self.split();
    img.warmup()?;
    txt.warmup()?;
    Ok(())
  }

  /// Zero-shot classification. Score is
  /// `sigmoid(exp(logit_scale)·cos + logit_bias) ∈ [0, 1]`.
  ///
  /// **The `labels` slice must contain fully-formed text prompts, not bare
  /// class names.** SigLIP2 was trained on web image-caption pairs, and the
  /// HuggingFace `pipeline("zero-shot-image-classification", model="google/siglip2-base-patch16-naflex")`
  /// usage convention wraps each label in a template like
  /// `"This is a photo of a {label}."` before encoding. Passing bare nouns
  /// (`"dog"`, `"cat"`, …) here will still produce well-ordered scores
  /// most of the time but the calibrated probabilities are noticeably
  /// worse than templated prompts on standard zero-shot benchmarks. We
  /// don't bake a template into this method because the optimal template
  /// is workload-dependent (`"a photo of {x}"` for natural images,
  /// `"a screenshot of {x}"` for UI captures, multilingual variants for
  /// non-English domains), and the same `LabeledScore` shape needs to
  /// flow through whatever the caller chose.
  ///
  /// Recommended call shape:
  ///
  /// ```ignore
  /// let prompts: Vec<String> = ["dog", "cat", "car"]
  ///     .iter()
  ///     .map(|c| format!("a photo of a {c}"))
  ///     .collect();
  /// let prompt_refs: Vec<&str> = prompts.iter().map(String::as_str).collect();
  /// let scored = siglip2.classify(image, &prompt_refs, 3)?;
  /// // scored[0].label() is the full prompt; strip the template if you
  /// // want to display the bare label.
  /// ```
  ///
  /// The calibration JSON's `logit_scale` field is the **raw learned
  /// parameter** (matching HuggingFace's `Siglip2Model.logit_scale`); the
  /// model exponentiates it at inference time. For the pinned release values
  /// (`logit_scale = 4.7476`, `logit_bias = -16.7770`), this produces an
  /// effective scale of `exp(4.7476) ≈ 115.36` — a typical confident match
  /// (cos ≈ 0.18) yields score ≈ 0.98, vs ~1e-7 if the exponentiation were
  /// skipped.
  ///
  /// `top_k` is clamped to `labels.len()`; passing a value larger than the
  /// label count returns all labels in descending score order rather than
  /// erroring.
  ///
  /// **Sort key is the cosine, not the sigmoid score.** With confident
  /// matches the f32 sigmoid output saturates to exactly `1.0` (this
  /// happens once `scale·cos + bias > ~16.6` in single precision
  /// for the pinned calibration that's any `cos > ~0.29`). Sorting
  /// by sigmoid would tie every saturated score and collapse the
  /// strongest matches' relative order to input order. Cosine has
  /// the full `[-1, 1]` f32 dynamic range and is monotone in the
  /// logit, so cosine descending = logit descending = what the user
  /// wants. The sigmoid scores returned in `LabeledScore::score`
  /// are computed from the (potentially saturated) logit and may
  /// tie at `1.0` even when the underlying ranking is well-defined.
  pub fn classify<'a>(
    &mut self,
    image: ImageView<'_>,
    labels: &[&'a str],
    top_k: usize,
  ) -> Result<Vec<LabeledScore<'a>>> {
    let calibration = self.calibration; // Calibration: Copy
    let (image_enc, text_enc) = self.split();
    let img_emb = image_enc.embed_pixels(image)?;
    let text_embs = text_enc.embed_batch(labels)?;
    let cosines: Vec<f32> = text_embs.iter().map(|t| img_emb.cosine(t)).collect();
    let ranked = rank_and_score(&cosines, calibration, top_k);
    Ok(
      ranked
        .into_iter()
        .map(|(i, _cos, score)| LabeledScore::new(labels[i], score))
        .collect(),
    )
  }
}

/// Rank label cosines and compute calibrated probabilities.
///
/// **Sort key is the cosine, not the sigmoid score.** The SigLIP2
/// pinned calibration (`exp(4.7476) ≈ 115`, `bias = -16.78`) makes
/// `sigmoid(scale·cos + bias)` round to exactly `1.0` in f32 once
/// `cos > ~0.29` — so any two labels with cosines above that
/// threshold tie under sigmoid sort, and `partial_cmp` collapses them
/// to input order, silently destroying ranking precision on the
/// most-confident matches. Cosine has the full f32 dynamic range
/// (`[-1, 1]`) and is monotone in the logit (effective scale is
/// always positive: `exp(logit_scale) > 0` for any finite scale), so
/// sorting by cosine descending = sorting by logit descending =
/// what the user means by "best matches first" — even when the
/// f32 sigmoid output ties.
///
/// Returns `(label_index, cosine, sigmoid_score)` triples in
/// descending cosine order, capped at `top_k`. The sigmoid output is
/// the user-visible `LabeledScore::score` and is computed AFTER the
/// sort, never as the sort key.
#[cfg_attr(not(tarpaulin), inline(always))]
fn rank_and_score(
  cosines: &[f32],
  calibration: Calibration,
  top_k: usize,
) -> Vec<(usize, f32, f32)> {
  let scale = calibration.logit_scale().exp();
  let bias = calibration.logit_bias();
  let mut scored: Vec<(usize, f32, f32)> = cosines
    .iter()
    .enumerate()
    .map(|(i, &cos)| {
      let logit = scale * cos + bias;
      let sig = 1.0 / (1.0 + (-logit).exp());
      (i, cos, sig)
    })
    .collect();
  scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
  scored.truncate(top_k.min(cosines.len()));
  scored
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn from_parts_rejects_invalid_calibration() {
    // We can't construct an `ort::Session` in a unit test, but we CAN exercise
    // the calibration-validation branch in isolation.
    let bad = Calibration::new(f32::NAN, 0.0);
    let res = Calibration::validate(bad.logit_scale(), bad.logit_bias());
    assert!(res.is_err());
  }

  /// with the pinned calibration
  /// (effective scale ≈ 115, bias ≈ -16.78), `sigmoid(scale·cos + bias)`
  /// rounds to exactly `1.0` in f32 once `cos > ~0.29`. If
  /// `rank_and_score` sorted by sigmoid, every label above that
  /// threshold would tie and the strongest matches' relative order
  /// would collapse to input order. This test pins the
  /// "sort-by-cosine, score-with-sigmoid" contract: feed a set of
  /// distinct cosines in the saturation regime and verify the
  /// returned order tracks cosine even though every sigmoid score is
  /// exactly 1.0.
  #[test]
  fn rank_and_score_orders_by_cosine_in_saturation_regime() {
    // Pinned release calibration values.
    let cal = Calibration::new(4.747_554_3, -16.776_989);
    // All cosines are well above the f32 sigmoid saturation
    // threshold (~0.29 with these constants), but distinct.
    let cosines = [0.30_f32, 0.55, 0.95, 0.42, 0.70];
    let ranked = rank_and_score(&cosines, cal, cosines.len());

    // Sanity: every score saturates to 1.0 — proves the sort key
    // CAN'T be the sigmoid (else this test couldn't disambiguate).
    for &(_, _, score) in &ranked {
      assert_eq!(score, 1.0, "expected f32 sigmoid saturation; got {score}");
    }

    // The order MUST track descending cosine, not input order.
    let order: Vec<usize> = ranked.iter().map(|&(i, _, _)| i).collect();
    // Input cosines: [0.30, 0.55, 0.95, 0.42, 0.70] at indices [0, 1, 2, 3, 4]
    // Descending cosine: 0.95 (i=2), 0.70 (i=4), 0.55 (i=1), 0.42 (i=3), 0.30 (i=0)
    assert_eq!(
      order,
      vec![2, 4, 1, 3, 0],
      "rank_and_score must order by cosine, not by saturated sigmoid"
    );
  }

  /// Sub-saturation regime: when cosines are below the saturation
  /// threshold, sigmoid is well-behaved and the cosine sort still
  /// matches what a sigmoid sort would have produced.
  #[test]
  fn rank_and_score_orders_by_cosine_in_normal_regime() {
    let cal = Calibration::new(4.747_554_3, -16.776_989);
    // Cosines below the saturation threshold; sigmoid scores will
    // be distinct.
    let cosines = [0.05_f32, 0.20, 0.10, 0.15];
    let ranked = rank_and_score(&cosines, cal, cosines.len());

    // Distinct sigmoid scores (the regime where the old behavior
    // would also have worked).
    let scores: Vec<f32> = ranked.iter().map(|&(_, _, s)| s).collect();
    for w in scores.windows(2) {
      assert!(
        w[0] >= w[1],
        "scores must descend (got {} then {})",
        w[0],
        w[1]
      );
    }

    // Order matches descending cosine.
    let order: Vec<usize> = ranked.iter().map(|&(i, _, _)| i).collect();
    assert_eq!(order, vec![1, 3, 2, 0]);
  }

  #[test]
  fn rank_and_score_respects_top_k() {
    let cal = Calibration::new(4.747_554_3, -16.776_989);
    let cosines = [0.05_f32, 0.20, 0.10, 0.15];
    let ranked = rank_and_score(&cosines, cal, 2);
    assert_eq!(ranked.len(), 2);
    let order: Vec<usize> = ranked.iter().map(|&(i, _, _)| i).collect();
    assert_eq!(order, vec![1, 3], "top-2 must be the two highest cosines");
  }
}
