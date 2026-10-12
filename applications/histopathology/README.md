# Evidence Field: complete runnable histopathology model

This package supplies the complete trained histopathology inference system: all three selected model states, the original learned optical/relative encoder, structural descriptors, signed classifier and covering recursion, the original independent reader, preprocessing and file-based inference. No training server or network connection is needed after installing dependencies.

## Image-upload interface

After installing the requirements, run:

```sh
python serve_histopathology.py
```

Open http://127.0.0.1:8766 in your browser. Upload a native 96x96 RGB lymph-node patch or choose a real supplied example. The application computes a prediction and displays the independently verified selected covers, signed margin maps, score decomposition and separate member probabilities. Images remain on the computer running this interface. This local research interface uses the existing model and does not alter its weights.

## Run on an image

Use Python 3.12. Extract the ZIP and open a terminal in this folder:

```sh
python -m pip install -r requirements.txt
python predict_histopathology.py --input examples/example_negative.png --output results/negative
python predict_histopathology.py --input examples/example_positive.png --output results/positive
python predict_histopathology.py --input examples/example_error.png --output results/error
```

For your own **native 96x96 RGB lymph-node patch**, replace the input filename. PNG, JPEG and TIFF are accepted if their dimensions and RGB mode satisfy the trained contract. Input conversion is float32 CHW RGB divided by 255. The unchanged model then applies its optical input adapter. The wrapper rejects incompatible images instead of silently changing their meaning. A whole slide or arbitrary disease image needs a separately validated pipeline; these checkpoints classify tumour presence in the central 32x32 window of a 96x96 lymph-node patch.

CPU is the default. `--device cuda` uses an available GPU. `--calibrated` selects the original source-calibrated probability mixture; both raw and calibrated outputs are always recorded. The published pathology headline uses raw probabilities. Each task prediction combines three member probabilities after softmax, not averaged logits.

## Every prediction contains

* `report.html`: self-contained visual result with input, each member's selected cover, supporting/opposing score map, signed contribution table and replay outcome.
* `evidence_overview.png`: a compact picture of the real input and all three members' covers and signed margins.
* `visible_evidence.json` and `signed_fields_seed_*.npz`: selected support coordinates, exact score allocations and their integral/reconstruction checks.
* `prediction.json`: predicted label, both probability modes, all member logits/probabilities, checkpoint hashes and input-preparation identity.
* `input.png` and `prepared_input.npy`: actual input and authenticated numerical input.
* Three `evidence_seed_*.json.gz` files: complete native descriptors, learned parameters, selected covering units, class scores and geometry.
* Three trusted-parameter sidecars bound to the original checkpoint/configuration.
* `independent_evidence_verification.json`: a fresh NumPy/SciPy process reconstructs all six class programs, selected complete covers, descriptors, signed score terms, optimal routes and ensemble probabilities. It does not import Torch.

The signed totals separate anchors, local-affine, parent-affine, member-product, parent-modulation and bias contributions. This is faithful computation accounting, not proof that the supports identify clinically meaningful cells. A new output directory is required so earlier decisions are preserved.

## Checked trained weights

Each member contains 10,550 learned parameters and all 14 state tensors; each `.pt` is 47,461 bytes. Architecture computations are supplied as code and are separate from learned state. All three local states match the independently downloaded original cloud archive and public model files. Full input-to-prediction inference was reproduced on 340 real external pathology-panel inputs with identical decisions. This panel is an audit, not the complete external evaluation.

Full-cohort reported pathology AUROC/accuracy remains 0.9356/84.46% and 0.9453/86.77% at CAMELYON17 centres 1/2. The source partition uses centres 0/3/4 and source-only checkpoint selection. Unequal comparator tuning and higher Gaussian/fixed-partition control point estimates are disclosed in the architecture manuscript. This package does not claim universal medical performance or clinical certification.

Three real fixed-panel examples are supplied, including an incorrect prediction. Their released labels and original probabilities are in `examples/Expected_Original_Outputs.json`; inference does not read these labels. See the linked repository for full split provenance and research records:
https://github.com/isararsiddique/evidence-field
https://huggingface.co/Isarar/Evidence-Field

`Package_Inventory.json` binds every immutable implementation, checkpoint, configuration and example by SHA-256. Third-party example data retain their original dataset conditions. No bulk training dataset, private records, credentials or cloud keys are included.

## Interpretation of the visual evidence

The hard covering program exports scores for selected supports. The viewer distributes each unit score uniformly across its support for display. Integrating the class field and adding bias reconstructs its member logit; subtracting the two class fields reconstructs that member's tumour-versus-normal margin. Red is a positive margin contribution and blue is an opposing contribution. Cover colours distinguish computational units. These displays are not validated cell annotations or pixel-causal attributions.

All three members are displayed. The ensemble applies softmax/temperature to each member and combines probabilities, so the individual score maps are not an additive explanation of the ensemble probability. Member disagreement is reported as a measured property, not as a validated clinical abstention rule.
