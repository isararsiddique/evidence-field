# Trained EF checkpoint functional verification

All six published weights were tested with the original implementation, using PyTorch 2.7.0 on CPU. Three independent members were evaluated for each task. No weights, thresholds, training recipes, or numerical modules were changed.

## Actual inference, not merely loading

| Test | Real inputs | Member-image inferences | Decisions matching the stored results |
|---|---:|---:|---:|
| CAMELYON17 held-out-hospital audit panel | 340 | 1,020 | 340/340, raw and calibrated |
| CT: complete LUNGx plus frozen LIDC examples | 85 | 255 | 85/85, raw and calibrated |

Maximum probability discrepancy was 4.95e-7, below the pre-existing 2e-5 tolerance. Every input was authenticated against its original prepared-input digest. CT member-logit maximum discrepancy was 2.04e-6. Probabilities varied across the real images; these are functioning classifiers, not constant outputs.

The 340 pathology patches are the existing class-balanced evidence audit panel, not the complete external cohorts. Its raw accuracy/AUROC is 75.00%/0.9084 (160 patches at hospital 1) and 82.78%/0.9211 (180 at hospital 2). These sampled-panel metrics do not replace the full-study results. The previously reported full-cohort raw metrics remain 84.46%/0.9356 and 86.77%/0.9453; this check did not re-run all 119,958 external patches.

All 73 LUNGx nodules were re-evaluated. Raw accuracy is 43/73 = 58.90%, AUROC 0.7215; source-calibrated accuracy is 44/73 = 60.27%, AUROC 0.7245. Calibration uses the unchanged source temperatures. Benign false positives remain substantial: raw 26/37; calibrated 24/37. The 12 LIDC examples are a frozen illustrative audit panel, not a complete or independent new cohort; their 12/12 agreement must not be generalized.

## Why the checkpoint is approximately 47 KB

Each member has 10,550 learned float32 parameters: 42,200 bytes. Its buffers add 96 bytes, giving 42,296 tensor bytes. PyTorch tensor metadata and serialization overhead bring the actual file to 47,461 bytes for histopathology or 47,729 bytes for CT. Each task uses three such states plus the original code, input adapter, configuration and recorded ensemble temperatures. Architecture operations are executable code; they are not all learned weights. Small checkpoint storage does not imply small intermediate memory or low training cost.

## Original cloud file checked

The authenticated private Spaces bundle `Evidence_Field/2026-10-12/132.145.131.192/publication_proof/Best_EF_Checkpoints.zip` was downloaded, all 63 archived payload hashes verified, and every checkpoint extracted independently. All six original cloud states have exactly the same SHA-256 as the local and public weights. The registered original bundle contains the same 47 KB-class states; it does not contain a missing larger replacement. Earlier experimental checkpoints remain separate from the source-selected primary models.

## Public model identity

The tested public download is https://huggingface.co/Isarar/Evidence-Field at commit `63477617bb040efb4dd086166eafda61a4642fdb`. Independent implementation and one real worked example are available at https://github.com/isararsiddique/evidence-field. The new functional receipts identify every checkpoint, input-panel digest and prediction-file digest. Fresh prediction arrays are retained locally with this report.

This is a reproduction and checkpoint-integrity result. It establishes that the authentic trained states execute and reproduce the recorded outputs, including errors. It does not establish clinical certification or diagnostic superiority.
