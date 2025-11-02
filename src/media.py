import hashlib
from io import BytesIO
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageOps
import pydicom

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn, mx = np.min(arr), np.max(arr)
    if mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = (arr - mn) / (mx - mn) * 255.0
    return arr.clip(0, 255).astype(np.uint8)

def load_preview_and_meta(filename: str, raw_bytes: bytes) -> Tuple[Image.Image, Dict[str, Any]]:
    """
    Returns (PIL preview image, metadata dict) for DICOM/PNG/JPG.
    Applies windowing if available; otherwise simple normalization.
    """
    h = sha256_hex(raw_bytes)
    meta: Dict[str, Any] = {"filename": filename, "sha256": h}

    # Try DICOM first
    try:
        ds = pydicom.dcmread(BytesIO(raw_bytes), force=True, stop_before_pixels=False)
        meta.update({
            "kind": "DICOM",
            "Modality": getattr(ds, "Modality", None),
            "PatientID": getattr(ds, "PatientID", None),
            "StudyInstanceUID": getattr(ds, "StudyInstanceUID", None),
            "SeriesInstanceUID": getattr(ds, "SeriesInstanceUID", None),
            "SOPInstanceUID": getattr(ds, "SOPInstanceUID", None),
            "StudyDate": getattr(ds, "StudyDate", None),
        })
        arr = ds.pixel_array  # type: ignore[attr-defined]

        # Windowing if present
        try:
            wc = ds.WindowCenter
            ww = ds.WindowWidth
            if hasattr(wc, "__len__"):
                wc = float(wc[0])
            else:
                wc = float(wc)
            if hasattr(ww, "__len__"):
                ww = float(ww[0])
            else:
                ww = float(ww)
            lo, hi = wc - ww/2.0, wc + ww/2.0
            arr = np.clip(arr, lo, hi)
        except Exception:
            pass

        prev = _normalize_to_uint8(arr)
        if prev.ndim == 2:
            img = Image.fromarray(prev, mode="L")
        elif prev.ndim == 3 and prev.shape[2] >= 3:
            img = Image.fromarray(prev[:, :, :3])
        else:
            # Fallback: single channel
            img = Image.fromarray(prev if prev.ndim == 2 else prev[:, :, 0])

        # Some chest X-rays are inverted; ensure darker = denser
        img = ImageOps.autocontrast(img)
        meta.update({"Rows": getattr(ds, "Rows", None), "Columns": getattr(ds, "Columns", None)})
        return img, meta
    except Exception:
        pass

    # Fall back to regular images
    try:
        im = Image.open(BytesIO(raw_bytes))
        meta.update({"kind": "Image", "Mode": im.mode, "Size": im.size})
        return ImageOps.autocontrast(im.convert("RGB")), meta
    except Exception:
        return Image.new("RGB", (256, 256), color=(20, 20, 20)), {**meta, "kind": "Unknown"}
