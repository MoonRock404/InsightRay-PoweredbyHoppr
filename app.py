import json
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.fhir import make_fhir_diag_report
from src.inference import (
    FINDING_MODELS, process_file, run_classifiers, run_vlm, compute_urgency,
    create_study, add_image, patient_label, patient_verdict, is_likely_normal,
    run_classifiers_with_payload, run_vlm_with_payload, extract_keywords
)

st.set_page_config(page_title="HOPPR Copilot", page_icon="🩻", layout="wide", initial_sidebar_state="expanded")
st.title("🩻 InsightRay: Powered by HOPPR")
st.caption("Built with the HOPPR Python SDK (`hopprai`).")

if "triage_rows" not in st.session_state:
    st.session_state.triage_rows = []

def color_dot(score: float) -> str:
    if score >= 0.70: return "🔴"
    if score >= 0.40: return "🟡"
    return "🟢"

# Sidebar
with st.sidebar:
    st.subheader("Choose Findings")
    active_models = {}
    for display_name, model_id in FINDING_MODELS.items():
        label = display_name if model_id else f"{display_name} (VLM only)"
        default_on = display_name in {"Pneumothorax", "Pleural Effusion", "Cardiomegaly"}
        if st.checkbox(label, value=default_on):
            active_models[display_name] = model_id  # may be None (VLM-only)
    st.subheader("Mode")
    mode = st.radio("View", ["Technician", "Patient"], horizontal=True)

# =============== Technician Mode ===============
if mode == "Technician":
    tab_queue, tab_single = st.tabs(["🧰 Triage Queue", "🔎 Single Case"])

    # ----- TRIAGE -----
    with tab_queue:
        st.subheader("Batch Upload & Triage")
        files = st.file_uploader("Upload multiple DICOM/PNG/JPG files",
                                 type=["dcm", "png", "jpg", "jpeg"], accept_multiple_files=True)
        col_a, col_b = st.columns([1,1])
        run_batch = col_a.button("Run Batch Inference", type="primary", use_container_width=True)
        clear_batch = col_b.button("Clear Queue", use_container_width=True)
        if clear_batch:
            st.session_state.triage_rows = []
        if run_batch:
            if not files:
                st.error("Please select one or more files.")
            elif not active_models:
                st.error("Select at least one finding to screen.")
            else:
                with st.spinner(f"Running {len(files)} file(s)…"):
                    for f in files:
                        row = process_file(f, active_models)
                        st.session_state.triage_rows.append(row)

        rows = st.session_state.triage_rows
        if rows:
            df = pd.DataFrame([{
                "Study ID": r["study_id"],
                "File": r["file"],
                "Urgency": r["urgency"],
                "Top Findings": r["top_summary"],
            } for r in rows]).sort_values("Urgency", ascending=False).reset_index(drop=True)
            st.dataframe(df, use_container_width=True)

            st.markdown("### Inspect Selected Case")
            chosen = st.selectbox("Pick a study", options=list(df["Study ID"]))
            r_map = {r["study_id"]: r for r in rows}
            if chosen in r_map:
                R = r_map[chosen]
                c1, c2 = st.columns([1,1])
                with c1:
                    st.markdown("**Scores**")
                    if R["scores"]:
                        fig, ax = plt.subplots()
                        names = list(R["scores"].keys()); vals = [R["scores"][k] for k in names]
                        ax.bar(names, vals); ax.set_ylim(0, 1); ax.set_ylabel("Probability (0–1)")
                        ax.set_xticklabels(names, rotation=20, ha="right")
                        st.pyplot(fig)
                        for k, v in sorted(R["scores"].items(), key=lambda kv: -kv[1]):
                            st.write(f"{color_dot(v)} **{k}** — {v:.2f}")
                    else:
                        st.info("No scores returned.")
                with c2:
                    badge = "🔴" if R["urgency"] >= 0.7 else ("🟡" if R["urgency"] >= 0.4 else "🟢")
                    st.metric("Urgency (weighted)", f"{badge} {R['urgency']:.2f}")
                    st.markdown("**Narrative (VLM)**"); st.write(R["vlm"] or "—")
                    fhir = make_fhir_diag_report(R["study_id"], R["scores"], R["vlm"])
                    colx, coly = st.columns([1,1])
                    colx.download_button("Download JSON",
                        data=json.dumps(R, indent=2), file_name=f"hoppr_result_{R['study_id']}.json",
                        mime="application/json", use_container_width=True)
                    coly.download_button("Download FHIR JSON",
                        data=json.dumps(fhir, indent=2), file_name=f"fhir_{R['study_id']}.json",
                        mime="application/json", use_container_width=True)
        else:
            st.info("No triage cases yet. Upload files and click **Run Batch Inference**.")

    # ----- SINGLE CASE -----
    with tab_single:
        st.subheader("Single Case")
        uploaded = st.file_uploader("Upload one DICOM/PNG/JPG",
                                    type=["dcm", "png", "jpg", "jpeg"], key="single_uploader")
        run_single = st.button("Run Inference (Single Case)", type="primary")
        if run_single:
            if not uploaded:
                st.error("Please upload a file first.")
            elif not active_models:
                st.error("Select at least one finding to screen.")
            else:
                try:
                    study_id = create_study(prefix="single")
                    add_image(study_id, uploaded.name, uploaded.read())
                except Exception as e:
                    st.error(f"Study/image failure: {e}")
                else:
                    with st.spinner("Running classifiers & VLM..."):
                        scores = run_classifiers(study_id, active_models)
                        vlm_text = run_vlm(study_id)
                        urgency = compute_urgency(scores)

                    c1, c2 = st.columns([1,1])
                    with c1:
                        st.markdown("**Scores**")
                        if scores:
                            fig, ax = plt.subplots()
                            names = list(scores.keys()); vals = [scores[k] for k in names]
                            ax.bar(names, vals); ax.set_ylim(0, 1); ax.set_ylabel("Probability (0–1)")
                            ax.set_xticklabels(names, rotation=20, ha="right")
                            st.pyplot(fig)
                            for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
                                st.write(f"{color_dot(v)} **{k}** — {v:.2f}")
                        else:
                            st.info("No scores returned.")
                    with c2:
                        badge = "🔴" if urgency >= 0.7 else ("🟡" if urgency >= 0.4 else "🟢")
                        st.metric("Urgency (weighted)", f"{badge} {urgency:.2f}")
                        st.markdown("**Narrative (VLM)**"); st.write(vlm_text or "—")

                        export_obj = {
                            "study_id": study_id, "scores": scores, "urgency": urgency,
                            "vlm_findings": vlm_text, "models": active_models,
                        }
                        colx, coly = st.columns([1,1])
                        colx.download_button("Download JSON",
                            data=json.dumps(export_obj, indent=2), file_name=f"hoppr_result_{study_id}.json",
                            mime="application/json", use_container_width=True)
                        fhir = make_fhir_diag_report(study_id, scores, vlm_text)
                        coly.download_button("Download FHIR JSON",
                            data=json.dumps(fhir, indent=2), file_name=f"fhir_{study_id}.json",
                            mime="application/json", use_container_width=True)

# =============== Patient Mode (polished single-case) ===============
else:
    st.subheader("Patient View")
    st.caption("Upload one image to see a friendly explanation.")

    # Threshold controls for clearer interpretation
    with st.expander("Display Settings (Patient)"):
        flag_thr = st.slider("Flag threshold (red)", 0.0, 1.0, 0.50, 0.01)
        maybe_thr = st.slider("Maybe threshold (amber, below red)", 0.0, flag_thr, 0.35, 0.01)
        st.caption("We color results by these thresholds. Adjust to be stricter/looser for demos.")

    uploaded = st.file_uploader("Upload one DICOM/PNG/JPG",
                                type=["dcm", "png", "jpg", "jpeg"], key="patient_uploader")
    run_patient = st.button("Run Inference (Patient)")

    if run_patient:
        if not uploaded:
            st.error("Please upload a file first.")
        else:
            try:
                study_id = create_study(prefix="patient")
                add_image(study_id, uploaded.name, uploaded.read())
            except Exception as e:
                st.error(f"Study/image failure: {e}")
            else:
                # Focused subset for patient (tweak as needed)
                patient_subset = {
                    k: FINDING_MODELS[k]
                    for k in ["Pneumothorax", "Pleural Effusion", "Cardiomegaly",
                              "Consolidation", "ILD"]
                    if k in FINDING_MODELS
                }

                with st.spinner("Analyzing your image..."):
                    scores, raw_payloads = run_classifiers_with_payload(study_id, patient_subset)
                    vlm_text, vlm_payload = run_vlm_with_payload(study_id)
                    vlm_hits = extract_keywords(vlm_text)

                # ---------- Summary ----------
                st.markdown("### Your Report:")
                st.write(vlm_text or "No notable concerns described.")

                if vlm_hits:
                    chips = " • ".join(f"`{k}`" for k in vlm_hits)
                    st.caption(f"Highlights from report: {chips}")

                # ---------- Key checks with progress bars ----------
                st.markdown("### Key Checks:xs")
                if scores:
                    if is_likely_normal(scores, threshold=maybe_thr):
                        st.success("Overall: likely no strong abnormalities detected.")

                    for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
                        verdict_txt, tone = patient_verdict(v)
                        # Override tone using chosen thresholds
                        if v >= flag_thr:
                            tone = "red"
                        elif v >= maybe_thr:
                            tone = "amber"
                        else:
                            tone = "green"
                        color = {"red": "#f87171", "amber": "#f59e0b", "green": "#34d399"}[tone]

                        label = patient_label(k)
                        box = st.container()
                        with box:
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(
                                    f"<div style='display:flex;justify-content:space-between;"
                                    f"align-items:center;border:1px solid #1f2937;border-radius:12px;"
                                    f"padding:10px;margin-bottom:6px'>"
                                    f"<span style='font-weight:600'>{label}</span>"
                                    f"<span style='color:{color}'>{verdict_txt}</span></div>",
                                    unsafe_allow_html=True
                                )
                                st.progress(min(max(v, 0.0), 1.0), text=f"Probability: {v:.2f}")
                            with c2:
                                st.write("")  # spacer
                                st.metric("Score", f"{v:.2f}")

                    st.caption("This view is educational and does not replace medical advice.")
                else:
                    st.info("No findings to display (classifier scores unavailable).")

                # ---------- Evidence for judges / debugging ----------
                with st.expander("🔎 Raw API Evidence (per-model payloads)"):
                    st.write("**Study ID:**", study_id)
                    st.json({"vlm": vlm_payload})
                    st.json({"classifiers": raw_payloads})

                st.caption(
                    "Tip: If everything stays low across many images, either your files are mostly normal, "
                    "or the chosen findings don’t match the pathology in your set. Try enabling other findings "
                    "(e.g., Opacity/Consolidation/ILD) or test with known positive examples."
                )
