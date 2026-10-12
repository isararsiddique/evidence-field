"""Portable reproduction of the worked EF example; no training or network."""
from pathlib import Path
import argparse, hashlib, importlib.util, json, sys
import numpy as np
from scipy.special import softmax

P = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_payloads():
    inventory = json.loads((P / "Reproduction_Inventory.json").read_text())
    for r in inventory["files"]:
        f = (P / r["file"]).resolve()
        assert f.is_relative_to(P.resolve()) and f.stat().st_size == r["bytes"] and sha(f) == r["sha256"], r["file"]
    return inventory

def load_helper():
    spec = importlib.util.spec_from_file_location("ef_read_only_helper", P / "reader/extended_evidence_diagnostics.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

def independent():
    inventory = verify_payloads()
    image = np.load(P / "example_input.npy", allow_pickle=False)
    h = load_helper()
    reader = h.reader_module(P / "model_code/locked_model_source")
    results = []; logits = []; weights = []; temps = []
    for seed in (11,29,47):
        trust = json.loads((P / f"trust_seed_{seed}.json").read_text())
        record = json.loads((P / f"example_seed_{seed}.json").read_text())
        assert sha(P / f"checkpoints/ef_two_ce__{seed}.pt") == trust["checkpoint_sha256"]
        assert np.isfinite(trust["temperature"]) and trust["temperature"] > 0
        assert np.isfinite(trust["weight"]) and trust["weight"] >= 0
        audit = h.extended_audit(record, image, trust, reader)
        assert audit["passed"] and max(audit["native_reader"][k] for k in
                                       ("descriptor_error","score_error","optimum_error")) <= 2e-5
        assert audit["maximum_decomposition_error"] <= 2e-5
        assert audit["native_reader"]["cover_voxels"] == [9216,9216]
        results.append(dict(seed=seed, audit=audit))
        logits.append([c["reconstructed_logit"] for c in audit["classes"]])
        weights.append(trust["weight"]); temps.append(trust["temperature"])
    z = np.array(logits); w=np.array(weights); t=np.array(temps)
    assert abs(w.sum()-1) < 1e-15
    raw=(softmax(z,axis=1)*w[:,None]).sum(0)
    calibrated=(softmax(z/t[:,None],axis=1)*w[:,None]).sum(0)
    expected=json.loads((P/"Worked_Evidence_Example.json").read_text())
    assert np.max(np.abs(raw-expected["raw_probability"])) <= 2e-5
    assert np.max(np.abs(calibrated-expected["calibrated_probability"])) <= 2e-5
    assert not any(n=="torch" or n.startswith("torch.") for n in sys.modules)
    return dict(passed=True, scope="One disclosed incorrect real patch, three members/six class programs",
                inventory_payloads_verified=len(inventory["files"]), independent_reader_framework_free=True,
                raw_probability=raw.tolist(), calibrated_probability=calibrated.tolist(),
                released_label=expected["released_label"], predicted_label=int(raw.argmax()), members=results)

def model_check():
    """Separate fresh process: strict original checkpoint/config loading."""
    verify_payloads()
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(P/"model_code/locked_model_source"))
    from evidence_field.dense_program import DenseOpticalProgram
    from evidence_field.stable_program import StableProgramConfig
    image=torch.from_numpy(np.load(P/"example_input.npy",allow_pickle=False))[None]
    logits=[]; probabilities=[]; rows=[]
    for seed in (11,29,47):
        trust=json.loads((P/f"trust_seed_{seed}.json").read_text())
        state=torch.load(P/f"checkpoints/ef_two_ce__{seed}.pt",map_location="cpu",weights_only=True)
        assert set(state)==set(trust["parameters"])
        assert all(np.array_equal(v.numpy(),np.asarray(trust["parameters"][k],dtype=v.numpy().dtype)) for k,v in state.items())
        model=DenseOpticalProgram(StableProgramConfig(**trust["config"])); model.load_state_dict(state,strict=True); model.eval()
        assert model.fingerprint()==trust["model_id"]
        with torch.no_grad():
            phi,g=model.descriptors(image)
            own,groups,_=model.scored(phi,g,0.)
            z=model.compose(own.double(),groups.double(),g,0.)[0].numpy()
        expected=np.asarray(json.loads((P/f"example_seed_{seed}.json").read_text())["logits"])
        err=float(np.max(np.abs(z-expected))); assert err<=2e-5
        rows.append(dict(seed=seed,strict_checkpoint_load=True,coefficients_match_trust=True,maximum_logit_error=err))
        logits.append(z);probabilities.append(trust["weight"]*softmax(z))
    p=np.sum(probabilities,axis=0)
    expected=json.loads((P/"Worked_Evidence_Example.json").read_text())
    assert np.max(np.abs(p-expected["raw_probability"]))<=2e-5
    return dict(passed=True,scope="CPU checkpoint inference on the same real patch; no optimizer steps",members=rows,
                raw_probability=p.tolist(),torch_version=torch.__version__)

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--with-model",action="store_true")
    args=p.parse_args();out=model_check() if args.with_model else independent()
    (P/("Model_Example_Reproduction.json" if args.with_model else "Reader_Example_Reproduction.json")).write_text(json.dumps(out,indent=2)+"\n")
    print(json.dumps({k:v for k,v in out.items() if k!="members"},indent=2))
