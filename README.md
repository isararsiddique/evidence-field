# Evidence Field

Research implementation and audited reproduction artifacts for **Evidence
Field: Adaptive Covering Programs for Medical Imaging**, Convolity AI Lab.

EF couples learned image-relative response descriptors, parent-conditioned
signed interactions and a finite covering dynamic program. Every member/class
exports its selected supports, scores, dependencies and optimum for independent
reconstruction. This contract does not establish clinical localization or
certification.

## Reproduce the real worked example

Clone this repository, then run with Python 3.12:

```sh
python download_artifacts.py
python -m pip install -r Evidence_Field_Reproduction/requirements-reader.txt
python Evidence_Field_Reproduction/reproduce_example.py
```

The default reader uses NumPy/SciPy and rejects Torch. It checks the descriptors,
scores, complete covers, routes, optima, signed accounting and all three member
probabilities. The disclosed patch has released central-window label 0; the
ensemble incorrectly predicts 1 with raw probability 0.88756339. Correct replay
and diagnostic correctness are separate outcomes.

For original checkpoint inference on CPU, additionally install `torch==2.7.0`
and run `python Evidence_Field_Reproduction/reproduce_example.py --with-model`.
This strictly loads all three real source-selected EF checkpoints, compares
every coefficient with the independent trust records and checks the example
logits. Neither command trains or uses external labels for model selection.

## Included and excluded

Readable `reproduction/` files expose the unchanged numerical source, reader,
configurations, all 18 primary training receipts, split procedures and provenance.
The compact archive also supplies three selected EF checkpoint files, the real
audit patch, all six class programs, trust records and saved primary/ablation
patient-bootstrap statistics. `artifacts/Manifest.json` authenticates the chunks
and decoded ZIP; the package inventory binds each payload by SHA256 and size.
Binary artifacts use explicit base64 text transport and the supplied decoder.

CAMELYON17 centres 0/3/4 supply source training/validation/test; centres 1/2 are
the two held-out hospitals within the same dataset. The exact public coded
patient groups, label counts and 455,954 retained patch count are recorded in
`reproduction/provenance/preparation_audit.json`. These centres had been examined
in earlier work: this is source-locked replication, not prospective validation
or the standard WILDS leaderboard protocol. Mirror origin and the distinction
between mirror-byte checks and original-publisher authentication are explicit.

Bulk images, five full sample-row CSVs, source-validation prediction arrays,
and full-cohort prediction/training archives are excluded. Their omitted-file
hashes are retained; full metadata remains in the separate full reproduction
delivery. The one-patch command reproduces an example, not every training run or
full-cohort statistic. No credentials, remote access keys or untested modules
are included. The CAMELYON data retain their original access/license conditions;
no new licence for third-party data is asserted here.

## Measured scope

Primary EF raw ensemble AUROC/accuracy is 0.9356/84.46% and 0.9453/86.77% at
the two held-out hospitals. The publication reports the tested from-scratch
ResNet18/ViT comparisons, unequal tuning and patient-level uncertainty. Gaussian
closure and fixed partitions obtain higher external AUROC point estimates than
primary EF, so learned-signature/adaptive-partition predictive benefits remain
unestablished. General 2D/3D mathematics does not imply 3D application validation.

Manuscripts are independent research drafts. No acceptance, clinical review,
institutional evaluation or global priority is asserted.


## Paper and complete working histopathology model

* [Reviewed architecture PDF](paper/Evidence_Field_Architecture.pdf) and [complete editable source](paper/Evidence_Field_Architecture_Source.zip).
* [Original model repository](https://huggingface.co/Isarar/Evidence-Field), including the tested [complete runnable histopathology application](https://huggingface.co/Isarar/Evidence-Field/resolve/main/Histopathology_EF_Ready_To_Run.zip). The application accepts a native 96x96 RGB lymph-node patch and emits predicted labels, raw/source-calibrated probabilities, all three member outputs and independently reconstructed signed evidence.
* [Functional verification](verification/Trained_Model_Functional_Verification.md): fresh inference reproduced 425/425 recorded decisions (340 real pathology audit patches and 85 CT crops), with maximum probability error below 5e-7. All six weights are identical to the independently downloaded original cloud bundle. The small state files contain 10,550 parameters each; the full encoder/classifier is code plus states.

The pathology audit panel is not the full external evaluation; its metrics and the previously completed full-study results are identified separately in the report. CT LUNGx performance remains limited despite correct checkpoint reproduction.

For direct image inference, download/extract the runnable ZIP, install its requirements and run:

```sh
python predict_histopathology.py --input examples/example_negative.png --output results/negative
```

Three real examples include an incorrect prediction. The supplied independent reader verifies every member/class covering, score decomposition and probability mixture in a separate NumPy/SciPy process. Input labels are not used for prediction.
