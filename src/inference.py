import json
import datetime as dt
import re
from typing import Dict, Any, Tuple, Optional, List
from .hoppr_client import get_hoppr, HOPPR

# ---- Models (direct, aliases, and VLM-only via None) ----
FINDING_MODELS: Dict[str, Optional[str]] = {
    # Direct classifiers
    "Atelectasis": "mc_chestradiography_atelectasis:v1.20250828",
    "Cardiomegaly": "mc_chestradiography_cardiomegaly:v1.20250828",
    "Pleural Effusion": "mc_chestradiography_pleural_effusion:v1.20250828",
    "Pneumothorax": "mc_chestradiography_pneumothorax:v1.20250828",

    # Aliases → closest available
    "Consolidation": "mc_chestradiography_air_space_opacity:v1.20250828",
    "Lung Opacity": "mc_chestradiography_air_space_opacity:v1.20250828",
    "Infiltration": "mc_chestradiography_air_space_opacity:v1.20250828",
    "ILD": "mc_chestradiography_interstitial_thickening:v1.20250828",
    "Pulmonary Fibrosis": "mc_chestradiography_interstitial_thickening:v1.20250828",

    # VLM-only for now (no shared classifier ids)
    "Aortic Enlargement": None,
    "Calcification": None,
    "Pleural Thickening": None,
    "Normal": None,
}

CRITICAL_FINDINGS = {"Pneumothorax", "Pleural Effusion"}
VLM_MODEL_ID = "cxr-vlm-experimental"

# ---------- response normalization helpers ----------
def _to_dict(maybe_obj: Any) -> Optional[Dict[str, Any]]:
    """Normalize HOPPR responses to dict whether dict, JSON string, or object.response."""
    if hasattr(maybe_obj, "response"):
        return _to_dict(getattr(maybe_obj, "response"))
    if isinstance(maybe_obj, dict):
        return maybe_obj
    if isinstance(maybe_obj, str):
        try:
            return json.loads(maybe_obj)
        except Exception:
            return None
    return None

def _extract_score(payload: Dict[str, Any]) -> Optional[float]:
    """Works for {'score': x} or {'response': {'score': x}}."""
    if "score" in payload and isinstance(payload["score"], (int, float)):
        return float(payload["score"])
    if "response" in payload and isinstance(payload["response"], dict):
        inner = payload["response"]
        if "score" in inner and isinstance(inner["score"], (int, float)):
            return float(inner["score"])
    return None
# -----------------------------------------------------

def extract_study_id(study_obj) -> str:
    if hasattr(study_obj, "id"):
        return study_obj.id  # type: ignore[attr-defined]
    if isinstance(study_obj, dict) and "id" in study_obj:
        return str(study_obj["id"])
    raise RuntimeError(f"Unexpected study response shape: {type(study_obj)} -> {study_obj!r}")

def create_study(prefix: str = "uiuc-mtchacks") -> str:
    hoppr = get_hoppr()
    s = hoppr.create_study(f"{prefix}-{dt.datetime.utcnow().strftime('%Y%m%d-%H%M%S-%f')}")
    return extract_study_id(s)

def add_image(study_id: str, filename: str, data: bytes) -> None:
    """Robust wrapper for different SDK signatures."""
    hoppr = get_hoppr()
    try:
        hoppr.add_study_image(study_id, filename, data)  # positional
        return
    except TypeError:
        pass
    try:
        hoppr.add_study_image(study_id=study_id, reference=filename, data=data)  # kw 'data'
        return
    except TypeError:
        pass
    try:
        hoppr.add_study_image(study_id=study_id, reference=filename, image_bytes=data)  # kw 'image_bytes'
        return
    except TypeError as e:
        raise RuntimeError(f"add_study_image signature not recognized: {e}")

# ---------- core inference (simple) ----------
def run_classifiers(study_id: str, models: Dict[str, Optional[str]]) -> Dict[str, float]:
    hoppr = get_hoppr()
    scores: Dict[str, float] = {}
    for name, model_id in models.items():
        if not model_id:
            continue
        try:
            resp = hoppr.prompt_model(
                study_id, model=model_id, prompt="ignored for classification", organization="hoppr"
            )
            payload = _to_dict(resp) or _to_dict(getattr(resp, "response", resp))
            if payload:
                s = _extract_score(payload)
                if s is not None:
                    scores[name] = s
        except Exception:
            continue
    return scores

def run_vlm(study_id: str) -> str:
    hoppr = get_hoppr()
    try:
        resp = hoppr.prompt_model(
            study_id,
            model=VLM_MODEL_ID,
            prompt="Provide a concise radiology-style description of key findings."
        )
        payload = _to_dict(resp) or _to_dict(getattr(resp, "response", resp))
        if isinstance(payload, dict):
            if "findings" in payload and isinstance(payload["findings"], str):
                return payload["findings"]
            if "response" in payload and isinstance(payload["response"], dict):
                inner = payload["response"]
                if "findings" in inner and isinstance(inner["findings"], str):
                    return inner["findings"]
        return ""
    except Exception:
        return ""

# ---------- debuggable inference (with raw payloads) ----------
def run_classifiers_with_payload(study_id: str, models: Dict[str, Optional[str]]) -> Tuple[Dict[str, float], Dict[str, Any]]:
    hoppr = get_hoppr()
    scores: Dict[str, float] = {}
    payloads: Dict[str, Any] = {}
    for name, model_id in models.items():
        if not model_id:
            continue
        try:
            resp = hoppr.prompt_model(
                study_id, model=model_id, prompt="ignored for classification", organization="hoppr"
            )
            payload = _to_dict(resp) or _to_dict(getattr(resp, "response", resp)) or {}
            s = _extract_score(payload) if isinstance(payload, dict) else None
            if s is not None:
                scores[name] = s
            payloads[name] = payload if payload else {"note": "empty or unparsable payload"}
        except Exception as e:
            payloads[name] = {"error": str(e)}
    return scores, payloads

def run_vlm_with_payload(study_id: str) -> Tuple[str, Dict[str, Any]]:
    hoppr = get_hoppr()
    try:
        resp = hoppr.prompt_model(
            study_id,
            model=VLM_MODEL_ID,
            prompt="Provide a concise radiology-style description of key findings."
        )
        payload = _to_dict(resp) or _to_dict(getattr(resp, "response", resp)) or {}
        findings = ""
        if isinstance(payload, dict):
            if "findings" in payload and isinstance(payload["findings"], str):
                findings = payload["findings"]
            elif "response" in payload and isinstance(payload["response"], dict):
                inner = payload["response"]
                if "findings" in inner and isinstance(inner["findings"], str):
                    findings = inner["findings"]
        return findings, (payload if payload else {"note": "empty or unparsable payload"})
    except Exception as e:
        return "", {"error": str(e)}

# ---------- scoring / presentation helpers ----------
def compute_urgency(scores: Dict[str, float]) -> float:
    vals = [(1.25 if k in CRITICAL_FINDINGS else 1.0) * v for k, v in scores.items()]
    return max(vals) if vals else 0.0

def patient_label(name: str) -> str:
    mapping = {
        "Pneumothorax": "Collapsed lung",
        "Pleural Effusion": "Fluid around the lungs",
        "Cardiomegaly": "Enlarged heart",
        "Lung Nodule or Mass": "Lung spot (nodule/mass)",
        "Consolidation": "Area of lung filled (consolidation)",
        "Lung Opacity": "Hazy area in lung (opacity)",
        "Infiltration": "Hazy area in lung (infiltration)",
        "ILD": "Scarring pattern (interstitial)",
        "Pulmonary Fibrosis": "Lung scarring (fibrosis)",
        "Aortic Enlargement": "Enlarged aorta",
        "Calcification": "Calcium deposits",
        "Pleural Thickening": "Thickened lining of lung",
        "Normal": "No clear abnormality",
    }
    return mapping.get(name, name)

def patient_verdict(score: float) -> Tuple[str, str]:
    if score >= 0.7:  return ("Needs prompt attention", "red")
    if score >= 0.4:  return ("Possibility present", "amber")
    return ("No strong signs", "green")

def is_likely_normal(scores: Dict[str, float], threshold: float = 0.30) -> bool:
    return bool(scores) and all(v < threshold for v in scores.values())

def extract_keywords(vlm_text: str) -> List[str]:
    """Simple keyword highlighter for patient view from VLM narrative."""
    if not vlm_text:
        return []
    keywords = [
        "pneumothorax", "effusion", "cardiomegaly", "opacity", "consolidation",
        "infiltrate", "interstitial", "fibrosis", "calcification", "pleural thickening",
        "aorta", "uncoiling", "enlarged", "nodule", "mass", "edema", "congestion"
    ]
    hits = []
    lt = vlm_text.lower()
    for k in keywords:
        try:
            if re.search(rf"\b{k}\b", lt):
                hits.append(k)
        except re.error:
            pass
    return sorted(set(hits))

# ---------- pipeline helpers ----------
def process_file(uploaded_file, models: Dict[str, Optional[str]]) -> Dict[str, Any]:
    study_id = create_study()
    add_image(study_id, uploaded_file.name, uploaded_file.read())
    scores = run_classifiers(study_id, models)
    urgency = compute_urgency(scores)
    vlm_text = run_vlm(study_id)
    top = sorted(scores.items(), key=lambda kv: -kv[1])[:3]
    top_summary = "; ".join([f"{k} {v:.2f}" for k, v in top]) if top else "—"
    return {"study_id": study_id, "file": uploaded_file.name, "urgency": urgency,
            "scores": scores, "top_summary": top_summary, "vlm": vlm_text}
