"""Bind actual cohort-level evaluation to an unchanged decision program.

This layer records a research evaluation's scope and integrity. It does not
turn an aggregate AUROC into a clinical certificate for an individual patient.
External evaluator attribution requires a supplied report, separate from the
in-house computational audit.
"""
from pathlib import Path
import hashlib,json
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def register_study(directory):
    directory=Path(directory);out=directory/'study_v01'
    result=json.loads((out/'training_results.json').read_text());audit=json.loads((out/'independent_audit.json').read_text())
    analysis=json.loads((out/'statistical_analysis.json').read_text());protocol=json.loads((out/'protocol.json').read_text())
    if not result.get('completed') or not audit.get('passed'):raise ValueError('Completed results and independent computational audit required')
    if result['source_lock_sha256']!=audit['source_lock_sha256']:raise ValueError('Audit state lock mismatch')
    if analysis['result_sha256']!=sha(out/'training_results.json') or analysis['audit_sha256']!=sha(out/'independent_audit.json'):raise ValueError('Analysis provenance mismatch')
    checkpoints={}
    for row in result['rows']:
        key=f'{row["method"]}__{row["seed"]}';path=out/(key+'.pt')
        if sha(path)!=row['checkpoint_sha256']:raise ValueError('Checkpoint identity mismatch')
        checkpoints[key]=row['checkpoint_sha256']
    report=dict(schema='ef-study-evidence/1.0',study=directory.name,
        scope='Cohort-level diagnostic evaluation and independent software audit; no individual clinical certification',
        execution_provenance='Convolity AI Lab computational study',
        third_party_evaluation=dict(status='report_pending',proposed_evaluator='Professor Sarinder Kaur',proposed_institution='University of Malaya'),
        source_lock_sha256=result['source_lock_sha256'],checkpoints=checkpoints,
        result_sha256=sha(out/'training_results.json'),audit_sha256=sha(out/'independent_audit.json'),analysis_sha256=sha(out/'statistical_analysis.json'),
        preparation_sha256=sha(directory/'prepared/preparation_audit.json'),protocol_sha256=sha(out/'protocol.json'),
        source_selected_families=analysis['source_selected_families'],cohort_results=analysis['pools'],
        precision=protocol.get('precision'),verified_records=audit['trained_evidence_records_verified'],
        maximum_descriptor_error=audit['maximum_descriptor_error'],maximum_score_error=audit['maximum_score_error'],maximum_optimum_error=audit['maximum_optimum_error'])
    report['report_id']=digest(report)
    (out/'study_evidence_registry.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report

def bind_study_evidence(candidate,receipt,report):
    """Attach verified study provenance without changing the original record."""
    expected=report['report_id'];body={k:v for k,v in report.items() if k!='report_id'}
    if digest(body)!=expected:raise ValueError('Study report digest mismatch')
    if not receipt.get('passed'):raise ValueError('Verified candidate receipt required')
    if receipt.get('input_hash')!=candidate['input_sha256'] or receipt.get('model_id')!=candidate['model_id']:raise ValueError('Candidate identity mismatch')
    key=f'{receipt["method"]}__{receipt["seed"]}'
    if report['checkpoints'].get(key)!=receipt['checkpoint_sha256']:raise ValueError('Candidate checkpoint not in evaluated study')
    return dict(schema='ef-evaluation-envelope/1.0',original_program=candidate,original_program_digest=digest(candidate),
        study_report_id=expected,study=report['study'],checkpoint_sha256=receipt['checkpoint_sha256'],
        case_receipt=receipt,cohort_scope=report['scope'],third_party_evaluation=report['third_party_evaluation'])

def verify_study_envelope(envelope,trusted_report,trusted_model_id):
    """Caller supplies trusted report and model identity; embedded hashes alone do not authenticate them."""
    if envelope['original_program']['model_id']!=trusted_model_id:raise ValueError('Externally trusted model mismatch')
    rebound=bind_study_evidence(envelope['original_program'],envelope['case_receipt'],trusted_report)
    if rebound!=envelope:raise ValueError('Envelope was altered')
    return dict(passed=True,original_program_preserved=True,study_report_id=trusted_report['report_id'],
        scope=trusted_report['scope'],third_party_status=trusted_report['third_party_evaluation']['status'])
