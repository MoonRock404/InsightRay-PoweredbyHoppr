from datetime import datetime
from typing import Dict, Any

def iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def make_fhir_diag_report(study_id: str, scores: Dict[str, float], vlm_text: str) -> Dict[str, Any]:
    observations = []
    for finding, score in scores.items():
        observations.append({
            "resourceType": "Observation",
            "status": "final",
            "code": {"text": finding},
            "valueQuantity": {"value": round(score, 3), "unit": "probability (0-1)"},
            "effectiveDateTime": iso_now(),
        })
    return {
        "resourceType": "DiagnosticReport",
        "status": "final",
        "category": [{"text": "Radiology"}],
        "code": {"text": "Chest radiograph AI assessment"},
        "effectiveDateTime": iso_now(),
        "conclusion": vlm_text,
        "result": observations,
        "presentedForm": [{
            "contentType": "application/json",
            "data": {"study_id": study_id}
        }],
    }
