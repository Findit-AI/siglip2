//! Integration tests gated on `SIGLIP2_MODELS_DIR`. Expand in Task 13 with
//! golden fixtures and parity assertions.

use std::path::PathBuf;

fn models_dir() -> Option<PathBuf> {
  std::env::var_os("SIGLIP2_MODELS_DIR").map(PathBuf::from)
}

#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn loads_image_encoder_from_release() {
  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
  let graph = dir.join("vision_model_naflex_256.onnx");
  let _enc = siglip2_naflex::ImageEncoder::from_files(&graph)
    .unwrap_or_else(|e| panic!("failed to load image encoder from {}: {e}", graph.display()));
}

#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn loads_text_encoder_from_release() {
  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
  let graph = dir.join("text_model_naflex.onnx");
  let tok = dir.join("tokenizer.json");
  let _enc = siglip2_naflex::TextEncoder::from_files(&graph, &tok)
    .unwrap_or_else(|e| panic!("failed to load text encoder: {e}"));
}

// Gated on `feature = "serde"`: `Calibration::from_path` parses the
// JSON via `serde_json` and is itself gated. Without serde, the
// equivalent path is `Calibration::bundled()` (covered by unit tests).
#[cfg(feature = "serde")]
#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn loads_calibration_from_release() {
  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
  let cal = siglip2_naflex::Calibration::from_path(&dir.join("calibration.json"))
    .expect("calibration must load");
  // Pinned to release values per spec §5.3.
  assert!((cal.logit_scale() - 4.747_554_3).abs() < 1e-3);
  assert!((cal.logit_bias() + 16.776_989).abs() < 1e-3);
}

#[test]
fn types_are_send_sync() {
  fn req<T: Send + Sync>() {}
  req::<siglip2_naflex::Preprocessor>();
  req::<siglip2_naflex::Embedding>();
  req::<siglip2_naflex::Calibration>();
}

// The `ort`-backed encoders are `Send` (`ort::Session` is `Send`). The MLX
// backend, which compiles unconditionally on Apple Silicon, embeds an
// `mlxrs::Siglip2NaflexModel` that holds `!Send` MLX device-array handles, so
// an MLX-backed encoder is single-threaded by construction. Assert `Send` only
// off Apple Silicon, where the `ort` backend is the only one compiled.
#[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
#[test]
fn encoders_are_send() {
  fn req<T: Send>() {}
  req::<siglip2_naflex::ImageEncoder>();
  req::<siglip2_naflex::TextEncoder>();
}

fn fixture_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn load_npy_f32_1d(path: &std::path::Path) -> Vec<f32> {
  let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
  let reader = npyz::NpyFile::new(&bytes[..]).unwrap();
  reader.into_vec::<f32>().unwrap()
}

// Parity is gated on SIGLIP2_MODELS_DIR (the released ONNX graphs) only.
// Golden fixtures under tests/fixtures/{images,embeddings} are committed,
// so the parity workflow no longer fails closed for fixture absence —
// see .github/workflows/parity.yml.
#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn image_parity_against_pytorch_reference() {
  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
  let mut enc = siglip2_naflex::ImageEncoder::from_files(&dir.join("vision_model_naflex_256.onnx"))
    .expect("encoder must load");

  let images_dir = fixture_dir().join("images");
  let embeddings_dir = fixture_dir().join("embeddings");
  let mut entries: Vec<_> = std::fs::read_dir(&images_dir)
    .unwrap_or_else(|e| panic!("fixture images missing at {}: {e}", images_dir.display()))
    .filter_map(|e| e.ok())
    .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
    .collect();
  entries.sort_by_key(|e| e.file_name());
  assert!(!entries.is_empty(), "no .png fixtures found");

  for entry in entries {
    let path = entry.path();
    let img = image::ImageReader::open(&path)
      .unwrap()
      .decode()
      .unwrap()
      .to_rgb8();
    let (w, h) = img.dimensions();
    let view = siglip2_naflex::ImageView::new(img.as_raw(), w, h).unwrap();
    let got = enc.embed_pixels(view).unwrap();

    let stem = path.file_stem().unwrap().to_string_lossy();
    let expected_path = embeddings_dir.join(format!("{stem}.npy"));
    let expected = load_npy_f32_1d(&expected_path);
    let expected_embedding = siglip2_naflex::Embedding::try_from(expected)
      .unwrap_or_else(|e| panic!("reference embedding for {stem} failed validation: {e}"));

    let cos = got.cosine(&expected_embedding);
    assert!(cos >= 0.99917, "{stem}: cosine {cos} below 0.99917 floor");
  }
}

#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn text_parity_against_pytorch_reference() {
  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
  let mut enc = siglip2_naflex::TextEncoder::from_files(
    &dir.join("text_model_naflex.onnx"),
    &dir.join("tokenizer.json"),
  )
  .expect("encoder must load");

  let prompts: Vec<String> =
    serde_json::from_slice(&std::fs::read(fixture_dir().join("text_prompts.json")).unwrap())
      .unwrap();
  let prompt_refs: Vec<&str> = prompts.iter().map(|s| s.as_str()).collect();

  let got = enc.embed_batch(&prompt_refs).unwrap();

  let raw = std::fs::read(fixture_dir().join("text_embeddings.npy")).unwrap();
  let reader = npyz::NpyFile::new(&raw[..]).unwrap();
  let shape = reader.shape().to_vec();
  let flat = reader.into_vec::<f32>().unwrap();
  assert_eq!(shape.len(), 2, "text_embeddings.npy must be 2-D");
  assert_eq!(
    shape[0] as usize,
    prompts.len(),
    "row count must match prompts.len()"
  );
  assert_eq!(shape[1], 768, "text embedding dim must be 768");

  for i in 0..prompts.len() {
    let row: Vec<f32> = flat[i * 768..(i + 1) * 768].to_vec();
    let expected = siglip2_naflex::Embedding::try_from(row).unwrap();
    let cos = got[i].cosine(&expected);
    assert!(
      cos >= 0.99917,
      "prompt {i} ({:?}): cosine {cos} below floor",
      prompts[i]
    );
  }
}

// `Siglip2::from_dir` must read the **directory's** `tokenizer.json`, not the
// embedded `BUNDLED_TOKENIZER`. The composed `from_onnx_dir` delegates its text
// side to `TextEncoder::from_onnx_dir(dir)` (which loads `dir/tokenizer.json`),
// so a `Siglip2` built from the directory must embed any prompt byte-identically
// to a standalone `TextEncoder` built from the same directory — proving the
// directory tokenizer (not the bundled one) is the one driving the text tower.
// Before the fix `Siglip2::from_onnx_dir` called `Self::bundled`, which loaded
// the embedded tokenizer and would silently diverge for a checkpoint shipping
// its own `tokenizer.json`. Gated on `feature = "bundled"` (the gate on
// `from_dir`) and `SIGLIP2_MODELS_DIR` (the released ONNX graphs + tokenizer).
#[cfg(feature = "bundled")]
#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn siglip2_from_dir_uses_directory_tokenizer() {
  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");

  // The standalone text encoder loads `dir/tokenizer.json` (its `from_onnx_dir`
  // path), giving the reference token-id behavior for this directory.
  let mut text_only =
    siglip2_naflex::TextEncoder::from_dir(&dir).expect("TextEncoder::from_dir must load");

  // The high-level wrapper must drive its text tower with that SAME directory
  // tokenizer — not the embedded `BUNDLED_TOKENIZER`.
  let mut s = siglip2_naflex::Siglip2::from_dir(&dir).expect("Siglip2::from_dir must load");

  // A mixed-case prompt exercises the lowercasing normalizer too; if the wrapper
  // were using a different tokenizer the embeddings would diverge.
  let prompt = "HELLO a photo of a cat";
  let from_wrapper = s
    .text()
    .embed(prompt)
    .expect("wrapper text embed must succeed");
  let from_standalone = text_only
    .embed(prompt)
    .expect("standalone text embed must succeed");

  let cos = from_wrapper.cosine(&from_standalone);
  assert!(
    cos >= 0.999_99,
    "Siglip2::from_dir must use the directory tokenizer (cosine vs standalone \
     TextEncoder::from_dir was {cos}, expected ~1.0)"
  );
}

// Uses `Siglip2::from_files`, which loads `calibration.json` via
// `serde_json` and is therefore gated on `feature = "serde"`. With
// `--features bundled` (no serde), JSON calibration loading is
// unavailable; the parity coverage of this test belongs in builds
// where serde is on.
#[cfg(feature = "serde")]
#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn cross_modal_ranking_sanity() {
  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
  let cal_path = dir.join("calibration.json");
  let mut s = siglip2_naflex::Siglip2::from_files(
    &dir.join("vision_model_naflex_256.onnx"),
    &dir.join("text_model_naflex.onnx"),
    &dir.join("tokenizer.json"),
    &cal_path,
  )
  .expect("siglip2 must load");

  // Use the first fixture image.
  let images_dir = fixture_dir().join("images");
  let first_png = std::fs::read_dir(&images_dir)
    .unwrap()
    .filter_map(|e| e.ok())
    .find(|e| e.path().extension().is_some_and(|x| x == "png"))
    .expect("at least one .png fixture")
    .path();
  let img = image::ImageReader::open(&first_png)
    .unwrap()
    .decode()
    .unwrap()
    .to_rgb8();
  let (w, h) = img.dimensions();
  let img_buf = img.into_raw();
  let view = siglip2_naflex::ImageView::new(&img_buf, w, h).unwrap();

  // Use templated prompts as the SigLIP2 zero-shot convention recommends
  // (`Siglip2::classify` docs explain why bare labels underperform).
  let labels = [
    "a photo of a sunset",
    "a screenshot of source code",
    "an MRI scan",
  ];
  let scored = s.classify(view, &labels, 3).expect("classify");
  assert_eq!(scored.len(), 3);
  // The fixture set is procedurally generated synthetic gradients, not real
  // photographs, so we can't pin which prompt wins. The descending-order
  // check is what the model contract guarantees on any input. A semantic
  // "intended label wins" assertion would need real-photo fixtures —
  // tracked separately if a richer corpus lands.
  for w in scored.windows(2) {
    assert!(
      w[0].score() >= w[1].score(),
      "classify must return descending score order"
    );
  }
}

/// Codex round-23 finding [high]: a caller can build a
/// `PreprocessedBatch` from a `Preprocessor` configured with one
/// `max_batch_size` and try to feed it to an `ImageEncoder` configured
/// with a smaller cap. The encoder must reject the oversized batch
/// with `Error::BatchTooLarge` rather than silently sending it to
/// ORT, otherwise the documented resource guard is meaningless on
/// the high-throughput path.
#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn embed_preprocessed_enforces_encoder_max_batch_size() {
  use siglip2_naflex::{BatchOptions, ImageView, Options, Preprocessor};

  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
  let graph = dir.join("vision_model_naflex_256.onnx");

  // Encoder configured with a tight cap (max_batch_size = 2).
  let small_opts = Options::default().with_batch(BatchOptions::default().with_max_batch_size(2));
  let mut enc = siglip2_naflex::ImageEncoder::from_files_with_options(&graph, small_opts)
    .expect("encoder must load");

  // Preprocessor configured with a looser cap (max_batch_size = 8).
  let big_opts = Options::default().with_batch(BatchOptions::default().with_max_batch_size(8));
  let pre = Preprocessor::new(big_opts).expect("preprocessor must build");

  // Build a batch of 4 images — within the preprocessor's cap, over
  // the encoder's cap.
  let rgb = vec![128u8; 32 * 32 * 3];
  let view = ImageView::new(&rgb, 32, 32).unwrap();
  let batch = pre
    .preprocess_batch(&[view, view, view, view])
    .expect("preprocess_batch within preprocessor's cap must succeed");
  assert_eq!(batch.len(), 4);

  // The encoder must reject. Without the round-23 fix this would
  // happily send 4 images into ORT despite max_batch_size = 2.
  let err = enc.embed_preprocessed(&batch).unwrap_err();
  match err {
    siglip2_naflex::Error::BatchTooLarge { got: 4, max: 2 } => {}
    _ => panic!("expected BatchTooLarge {{ got: 4, max: 2 }}, got {err}"),
  }
}

/// Codex round-20 finding [medium]: when `embed_batch` finds an
/// empty string in a multi-element input, the error must be
/// `Error::Batch { index, source: EmptyText }` so callers can
/// identify and replace the bad record. Previous behavior returned
/// bare `Error::EmptyText` and dropped the index — fine for the
/// single-item `embed("")` case but lossy for batch / classify
/// callers.
#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn embed_batch_empty_text_surfaces_index() {
  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
  let mut enc = siglip2_naflex::TextEncoder::from_files(
    &dir.join("text_model_naflex.onnx"),
    &dir.join("tokenizer.json"),
  )
  .expect("encoder must load");

  // Position 2 is empty; positions 0, 1, 3 are valid prompts.
  let texts = ["a photo of a sunset", "a screenshot", "", "an MRI scan"];
  let err = enc.embed_batch(&texts).unwrap_err();
  match err {
    siglip2_naflex::Error::Batch { index, source } => {
      assert_eq!(index, 2, "expected the empty-string index, got {index}");
      assert!(
        matches!(*source, siglip2_naflex::Error::EmptyText),
        "expected source to be EmptyText, got {source}"
      );
    }
    _ => panic!("expected Error::Batch, got {err}"),
  }
}

/// An oversized batch that *also* contains an empty string must return
/// `Error::BatchTooLarge` (the batch-shape problem), NOT
/// `Error::Batch { source: EmptyText }`. The max-batch cap is checked before
/// the per-item empty scan in the outer `embed_batch`, so the batch-size
/// failure wins — restoring the historical ONNX-path ordering (empty slice →
/// max-batch cap → per-item empty scan → dispatch).
#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn embed_batch_oversized_with_empty_reports_too_large_first() {
  use siglip2_naflex::{BatchOptions, Options};

  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
  // Tight cap: max_batch_size = 2.
  let opts = Options::default().with_batch(BatchOptions::default().with_max_batch_size(2));
  let mut enc = siglip2_naflex::TextEncoder::from_files_with_options(
    &dir.join("text_model_naflex.onnx"),
    &dir.join("tokenizer.json"),
    opts,
  )
  .expect("encoder must load");

  // 4 items (> cap of 2) AND position 1 is empty. The max-batch check must win.
  let texts = ["a photo of a sunset", "", "a screenshot", "an MRI scan"];
  let err = enc.embed_batch(&texts).unwrap_err();
  match err {
    siglip2_naflex::Error::BatchTooLarge { got: 4, max: 2 } => {}
    _ => panic!("expected BatchTooLarge {{ got: 4, max: 2 }} (not Batch/EmptyText), got {err}"),
  }
}

/// Codex round-28 finding [medium]: `TextEncoder` constructors must
/// apply the same `BatchOptions::validate` rules `Preprocessor::new`
/// does. Previously `from_ort_session_with_options` only ran
/// session-shape checks and stored opts, so a `batch_size = 0` config
/// was accepted at construction and silently coerced to `1` inside
/// `embed_batch` — surfacing as latent throughput loss instead of an
/// explicit construction error.
#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn text_encoder_rejects_zero_batch_size() {
  use siglip2_naflex::{BatchOptions, Options};

  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
  let bad = Options::default().with_batch(BatchOptions::default().with_batch_size(0));
  // `expect_err` requires `T: Debug` which `TextEncoder` deliberately
  // does not implement (the inner `ort::Session` is not `Debug`).
  // Use `.err().expect(...)` to keep the assertion message while
  // dropping the unused success value.
  let err = siglip2_naflex::TextEncoder::from_files_with_options(
    &dir.join("text_model_naflex.onnx"),
    &dir.join("tokenizer.json"),
    bad,
  )
  .err()
  .expect("batch_size = 0 must be rejected at construction, not silently coerced");
  match err {
    siglip2_naflex::Error::InvalidBatchSize {
      batch_size: 0,
      max_batch_size: 1024,
    } => {}
    _ => panic!("expected InvalidBatchSize {{ 0, 1024 }}, got {err}"),
  }
}

#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn text_encoder_rejects_batch_size_above_max() {
  use siglip2_naflex::{BatchOptions, Options};

  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
  let bad = Options::default().with_batch(
    BatchOptions::default()
      .with_batch_size(2048)
      .with_max_batch_size(1024),
  );
  let err = siglip2_naflex::TextEncoder::from_files_with_options(
    &dir.join("text_model_naflex.onnx"),
    &dir.join("tokenizer.json"),
    bad,
  )
  .err()
  .expect("batch_size > max_batch_size must be rejected at construction");
  match err {
    siglip2_naflex::Error::InvalidBatchSize {
      batch_size: 2048,
      max_batch_size: 1024,
    } => {}
    _ => panic!("expected InvalidBatchSize {{ 2048, 1024 }}, got {err}"),
  }
}

/// Single-item `embed("")` keeps the bare `Error::EmptyText` shape
/// (no batch index makes sense for a single input).
#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn embed_single_empty_text_returns_empty_text() {
  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
  let mut enc = siglip2_naflex::TextEncoder::from_files(
    &dir.join("text_model_naflex.onnx"),
    &dir.join("tokenizer.json"),
  )
  .expect("encoder must load");

  let err = enc.embed("").unwrap_err();
  match err {
    siglip2_naflex::Error::EmptyText => {}
    _ => panic!("expected Error::EmptyText, got {err}"),
  }
}

/// Codex round-18 follow-up: pin the architectural contract that
/// `embed_pixels_batch(views)[i]` returns the same embedding (within
/// f32 rounding) as `embed_pixels(views[i])`. SigLIP2's vision tower
/// has no cross-image computation — self-attention is per-sequence,
/// LayerNorm is per-token, padded patch slots are mask-excluded, so
/// each image's output depends only on its own input. ORT may dispatch
/// different GEMM tile sizes for `batch=1` vs `batch=N`, producing
/// tiny f32 reordering differences in reductions, but the cosine
/// between single and batched outputs must stay well above any
/// retrieval threshold (we assert ≥ 0.99999 — five nines, about three
/// orders of magnitude tighter than the PyTorch parity floor).
#[test]
#[ignore = "requires SIGLIP2_MODELS_DIR"]
fn batched_embedding_matches_single_image() {
  let dir = models_dir().expect("SIGLIP2_MODELS_DIR not set");
  let mut enc = siglip2_naflex::ImageEncoder::from_files(&dir.join("vision_model_naflex_256.onnx"))
    .expect("encoder must load");

  // Use the fixture images. Decode each into an owned RGB buffer so
  // the borrows live long enough to hand `&[ImageView]` to
  // `embed_pixels_batch`.
  let images_dir = fixture_dir().join("images");
  let mut entries: Vec<_> = std::fs::read_dir(&images_dir)
    .unwrap_or_else(|e| panic!("fixture images missing at {}: {e}", images_dir.display()))
    .filter_map(|e| e.ok())
    .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
    .collect();
  entries.sort_by_key(|e| e.file_name());
  assert!(!entries.is_empty(), "no .png fixtures found");

  // Decode all images first so the buffers outlive both the per-image
  // and batched embedding calls.
  let decoded: Vec<(Vec<u8>, u32, u32, String)> = entries
    .iter()
    .map(|entry| {
      let path = entry.path();
      let img = image::ImageReader::open(&path)
        .unwrap()
        .decode()
        .unwrap()
        .to_rgb8();
      let (w, h) = img.dimensions();
      let stem = path.file_stem().unwrap().to_string_lossy().into_owned();
      (img.into_raw(), w, h, stem)
    })
    .collect();
  let views: Vec<siglip2_naflex::ImageView<'_>> = decoded
    .iter()
    .map(|(buf, w, h, _)| siglip2_naflex::ImageView::new(buf, *w, *h).unwrap())
    .collect();

  // Single-image path.
  let singles: Vec<siglip2_naflex::Embedding> = views
    .iter()
    .map(|v| enc.embed_pixels(*v).expect("embed_pixels"))
    .collect();

  // Batched path.
  let batched = enc.embed_pixels_batch(&views).expect("embed_pixels_batch");
  assert_eq!(
    batched.len(),
    singles.len(),
    "batched and single counts must match"
  );

  // Per-image cosine — each pair must be five-nines close.
  // SigLIP2's vision tower is per-sequence, so the only delta vs
  // single-image inference is f32 reordering inside ORT's GEMM.
  for (i, ((s, b), (_, _, _, stem))) in singles
    .iter()
    .zip(batched.iter())
    .zip(decoded.iter())
    .enumerate()
  {
    let cos = s.cosine(b);
    assert!(
      cos >= 0.99999,
      "image {i} ({stem}): batched vs single cosine {cos} below 0.99999 \
       — batching is influencing per-image output beyond f32 rounding"
    );
  }
}

// ───────────────────────── MLX backend parity ──────────────────────────
//
// Runs the committed PyTorch-reference fixture images through the MLX
// (`mlxrs`) backend and compares each embedding to the same reference `.npy`
// the ORT parity test uses. Gated on Apple Silicon (the only place the backend
// compiles) and on `SIGLIP2_MLX_DIR` pointing at an MLX checkpoint directory
// (`config.json` + `model.safetensors` + `tokenizer.json`); `from_dir`
// auto-routes that directory to MLX. The MLX path resamples with a PIL-bit-exact
// bilinear (the upstream HF processor's filter), so it is expected to track the
// reference at least as closely as the ORT path's `image`-rs `Triangle` resize.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires SIGLIP2_MLX_DIR (an MLX checkpoint directory)"]
fn mlx_image_parity_against_pytorch_reference() {
  let dir = std::env::var_os("SIGLIP2_MLX_DIR")
    .map(PathBuf::from)
    .expect("SIGLIP2_MLX_DIR not set");
  // `from_dir` auto-routes to MLX on Apple Silicon when `SIGLIP2_MLX_DIR` holds
  // an MLX checkpoint (`config.json` + `model.safetensors`) — there is no
  // public MLX-specific constructor; the platform picks the backend.
  let mut enc = siglip2_naflex::ImageEncoder::from_dir(&dir).unwrap_or_else(|e| {
    panic!(
      "failed to load MLX image encoder from {}: {e}",
      dir.display()
    )
  });

  let images_dir = fixture_dir().join("images");
  let embeddings_dir = fixture_dir().join("embeddings");
  let mut entries: Vec<_> = std::fs::read_dir(&images_dir)
    .unwrap_or_else(|e| panic!("fixture images missing at {}: {e}", images_dir.display()))
    .filter_map(|e| e.ok())
    .filter(|e| e.path().extension().is_some_and(|x| x == "png"))
    .collect();
  entries.sort_by_key(|e| e.file_name());
  assert!(!entries.is_empty(), "no .png fixtures found");

  let mut min_cos = f32::INFINITY;
  for entry in entries {
    let path = entry.path();
    let img = image::ImageReader::open(&path)
      .unwrap()
      .decode()
      .unwrap()
      .to_rgb8();
    let (w, h) = img.dimensions();
    let view = siglip2_naflex::ImageView::new(img.as_raw(), w, h).unwrap();
    let got = enc.embed_pixels(view).unwrap();

    let stem = path.file_stem().unwrap().to_string_lossy();
    let expected = load_npy_f32_1d(&embeddings_dir.join(format!("{stem}.npy")));
    let expected_embedding = siglip2_naflex::Embedding::try_from(expected)
      .unwrap_or_else(|e| panic!("reference embedding for {stem} failed validation: {e}"));

    let cos = got.cosine(&expected_embedding);
    min_cos = min_cos.min(cos);
    // The MLX path's resize filter is the upstream PIL-bit-exact bilinear (the
    // reference HF processor's filter), so it tracks the PyTorch reference far
    // more tightly than the ORT path's `image`-rs `Triangle` (documented
    // 0.99917 floor): the measured min cosine across the fixture set is
    // ~0.9999999. This 0.9995 floor is a comfortable regression guard.
    assert!(
      cos >= 0.9995,
      "{stem}: MLX-vs-PyTorch cosine {cos} below 0.9995 floor"
    );
  }
  eprintln!("MLX image parity: min cosine vs PyTorch reference = {min_cos}");
}

// End-to-end text parity for the MLX backend: a MIXED-CASE prompt and its
// lowercase form must embed identically (the SigLIP2 lowercasing normalizer the
// serialized tokenizer.json lacks is installed by `prepare_mlx_tokenizer`).
// Exercises the full GPU text tower, complementing the GPU-free input_ids unit
// tests in `src/text_enc.rs`.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires SIGLIP2_MLX_DIR (an MLX checkpoint directory)"]
fn mlx_text_mixed_case_embeds_identically() {
  let dir = std::env::var_os("SIGLIP2_MLX_DIR")
    .map(PathBuf::from)
    .expect("SIGLIP2_MLX_DIR not set");
  // `from_dir` auto-routes to MLX on Apple Silicon when `SIGLIP2_MLX_DIR` holds
  // an MLX checkpoint — there is no public MLX-specific constructor.
  let mut enc = siglip2_naflex::TextEncoder::from_dir(&dir).unwrap_or_else(|e| {
    panic!(
      "failed to load MLX text encoder from {}: {e}",
      dir.display()
    )
  });

  let upper = enc.embed("A PHOTO OF A CAT").expect("embed upper");
  let lower = enc.embed("a photo of a cat").expect("embed lower");
  let cos = upper.cosine(&lower);
  assert!(
    cos >= 0.99999,
    "mixed-case and lowercase prompts must embed identically (cosine {cos}); \
     the SigLIP2 lowercasing normalizer is missing"
  );
}

// The MLX TEXT batch path rejects an oversized batch with the typed
// `BatchTooLarge` (the same contract as the ORT path) BEFORE allocating /
// running — not a panic / OOM. The crate-default cap is 1024, so 1025 entries
// trip it; the strings are never tokenized because the cap check is first.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires SIGLIP2_MLX_DIR (an MLX checkpoint directory)"]
fn mlx_text_batch_rejects_oversized() {
  let dir = std::env::var_os("SIGLIP2_MLX_DIR")
    .map(PathBuf::from)
    .expect("SIGLIP2_MLX_DIR not set");
  // `from_dir` auto-routes to MLX on Apple Silicon when `SIGLIP2_MLX_DIR` holds
  // an MLX checkpoint — there is no public MLX-specific constructor.
  let mut enc = siglip2_naflex::TextEncoder::from_dir(&dir).unwrap_or_else(|e| {
    panic!(
      "failed to load MLX text encoder from {}: {e}",
      dir.display()
    )
  });

  let over = 1025usize; // default max_batch_size is 1024
  let texts: Vec<&str> = vec!["x"; over];
  match enc.embed_batch(&texts) {
    Err(siglip2_naflex::Error::BatchTooLarge { got, max }) => {
      assert_eq!(got, over);
      assert_eq!(max, 1024);
    }
    other => panic!("expected BatchTooLarge {{ got: {over}, max: 1024 }}, got {other:?}"),
  }
}

// The MLX IMAGE batch path rejects an oversized batch with the typed
// `BatchTooLarge` BEFORE allocating / running, symmetric with the text path and
// the ORT path. The 1025 `ImageView`s share one tiny RGB buffer; the cap check
// runs before any preprocessing / GPU work.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
#[test]
#[ignore = "requires SIGLIP2_MLX_DIR (an MLX checkpoint directory)"]
fn mlx_image_batch_rejects_oversized() {
  let dir = std::env::var_os("SIGLIP2_MLX_DIR")
    .map(PathBuf::from)
    .expect("SIGLIP2_MLX_DIR not set");
  // `from_dir` auto-routes to MLX on Apple Silicon when `SIGLIP2_MLX_DIR` holds
  // an MLX checkpoint (`config.json` + `model.safetensors`) — there is no
  // public MLX-specific constructor; the platform picks the backend.
  let mut enc = siglip2_naflex::ImageEncoder::from_dir(&dir).unwrap_or_else(|e| {
    panic!(
      "failed to load MLX image encoder from {}: {e}",
      dir.display()
    )
  });

  let rgb = vec![128u8; 8 * 8 * 3];
  let view = siglip2_naflex::ImageView::new(&rgb, 8, 8).expect("view");
  let over = 1025usize;
  let views: Vec<siglip2_naflex::ImageView<'_>> = vec![view; over];
  match enc.embed_pixels_batch(&views) {
    Err(siglip2_naflex::Error::BatchTooLarge { got, max }) => {
      assert_eq!(got, over);
      assert_eq!(max, 1024);
    }
    other => panic!("expected BatchTooLarge {{ got: {over}, max: 1024 }}, got {other:?}"),
  }
}
