# imports
import glob
import json
import os
import re
from typing import Iterable, Optional, Dict, Any, List, Tuple

import bct
import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import datasets, maskers
from nilearn.connectome import ConnectivityMeasure


def _normalize_subject_id(raw_id: str) -> str:
    """Ensure subject ID follows BIDS format (e.g., 'PC001' -> 'sub-PC001')."""
    clean_id = raw_id.strip()
    return clean_id if clean_id.startswith("sub-") else f"sub-{clean_id}"


# 1. DEMOGRAPHICS, DOSING, AND PSYCHOMETRIC QUESTIONNAIRES (SES-01 & SES-02)
def extract_patient_clinical_demo(
    patient_files: Iterable[str],
    subject_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract age, psilocybin dose, and scored psychometric scales for ses-01 and ses-02."""
    files_list = list(patient_files)

    # Detect subject ID from file names if not explicitly provided
    if subject_id is None:
        for path in files_list:
            match = re.search(r"(sub-PC\d+)", os.path.basename(path))
            if match:
                subject_id = match.group(1)
                break
    if subject_id is None:
        raise ValueError("Could not detect 'sub-PCXXX' in file names. Pass subject_id explicitly.")

    subject_id = _normalize_subject_id(subject_id)

    # Load a single subject row from a BIDS TSV table (handling 'nd' and 'n/a' missing codes)
    def _load_subject_row(file_path: Optional[str]) -> Optional[pd.Series]:
        if not file_path or not os.path.exists(file_path):
            return None
        df = pd.read_csv(file_path, sep="\t", na_values=["nd", "n/a", "NA", "null", ""])
        id_col = next((c for c in df.columns if c.lower() in ["participant_id", "subject_id", "sub_id"]), None)
        if id_col is None:
            return None
        normalized_col = df[id_col].astype(str).apply(_normalize_subject_id)
        matched = df[normalized_col == subject_id]
        return matched.iloc[0] if not matched.empty else None

    # Match participants.tsv and participants.json
    participants_tsv_path = next((p for p in files_list if os.path.basename(p) == "participants.tsv"), None)
    participants_json_path = next((p for p in files_list if os.path.basename(p) == "participants.json"), None)

    # Validate schema fields against participants.json
    allowed_fields = {"participant_id", "age", "dose_mg_per_kg"}
    if participants_json_path and os.path.exists(participants_json_path):
        with open(participants_json_path, "r", encoding="utf-8") as f:
            participants_meta = json.load(f)
        allowed_fields = allowed_fields.intersection(participants_meta.keys())

    row_part = _load_subject_row(participants_tsv_path)

    def _get_part_val(col_name: str, default=np.nan):
        if row_part is not None and col_name in row_part.index and pd.notna(row_part[col_name]):
            return row_part[col_name]
        return default

    age = pd.to_numeric(_get_part_val("age"), errors="coerce") if "age" in allowed_fields else np.nan
    dose_mg_per_kg = (
        pd.to_numeric(_get_part_val("dose_mg_per_kg"), errors="coerce")
        if "dose_mg_per_kg" in allowed_fields
        else np.nan
    )

    result: Dict[str, Any] = {
        "participant_id": subject_id,
        "age": float(age) if pd.notna(age) else np.nan,
        "dose_mg_per_kg": float(dose_mg_per_kg) if pd.notna(dose_mg_per_kg) else np.nan,
    }

    # Extract psychometric questionnaires for ses-01 (baseline) and ses-02 (psilocybin)
    ignore_cols = {
        "participant_id", "psi_id", "ses", "session", "edi_one",
        "session_id", "visitday", "redcap_event_name", "imputation_threshold"       # to check threshold
    }
    session_questionnaires = {
        "ses-01": [("asc11", "ASC11.TSV"), ("idas2", "IDAS-II.TSV")],
        "ses-02": [("edi", "EDI.TSV"), ("meq30", "MEQ30.TSV"), ("asc11", "ASC11.TSV")],
    }

    for ses, q_list in session_questionnaires.items():
        ses_tag = ses.replace("-", "")
        for prefix, target_fname in q_list:
            q_path = next(
                (
                    p for p in files_list
                    if os.path.basename(p).upper() == target_fname
                    and ses in os.path.normpath(p).split(os.sep)
                ),
                None,
            )
            q_row = _load_subject_row(q_path)
            if q_row is not None:
                for col, val in q_row.items():
                    if col.lower() in ignore_cols:
                        continue
                    num_val = pd.to_numeric(val, errors="coerce")
                    clean_col = col if col.lower().startswith(prefix) else f"{prefix}_{col}"
                    result[f"{ses_tag}_{clean_col}"] = float(num_val) if pd.notna(num_val) else val

    return result


# 2. SCHAEFER 200 ATLAS (7 YEO FUNCTIONAL NETWORKS)
def load_schaefer_atlas(n_rois: int = 200, yeo_networks: int = 7) -> Tuple[str, np.ndarray, Dict[str, int]]:
    """
    Fetch the Schaefer 200 cortical parcellation atlas (7 Yeo networks) in MNI space
    and map each of the 200 ROIs (excluding 'Background') to its Yeo module ID.
    """
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=yeo_networks)
    raw_labels = [lbl.decode("utf-8") if isinstance(lbl, bytes) else str(lbl) for lbl in atlas.labels]

    # Filter out 'Background' label added by newer nilearn versions so length matches n_rois (200)
    roi_labels = [lbl for lbl in raw_labels if lbl != "Background" and len(lbl.split("_")) >= 3]
    if len(roi_labels) != n_rois:
        raise ValueError(f"Expected {n_rois} ROI labels, got {len(roi_labels)}.")

    # Label format: '7Networks_LH_Default_PFC_1' -> network name is at index 2
    network_names = [lbl.split("_")[2] for lbl in roi_labels]
    unique_networks = sorted(list(set(network_names)))
    net_to_id = {name: idx + 1 for idx, name in enumerate(unique_networks)}
    ci_yeo = np.array([net_to_id[name] for name in network_names], dtype=int)

    return atlas.maps, ci_yeo, net_to_id



# 3. CONFOUND TIMESERIES & HEAD MOTION QUALITY CONTROL (QC)
def extract_confounds_and_qc(
    base_dir: str,
    subject_id: str,
    session: str,
    task: str = "rest",
    strategy: str = "friston24_acompcor",
    fd_threshold: float = 0.5,
    scrubbing: bool = True,
) -> Tuple[Optional[pd.DataFrame], Dict[str, Any]]:
    """
    Parse fMRIPrep desc-confounds_timeseries.tsv to compute scalar motion QC metrics
    and build a clean nuisance regressor matrix for NiftiLabelsMasker denoising.
    """
    sub_id = _normalize_subject_id(subject_id)
    ses_tag = session.replace("-", "")
    prefix = f"{ses_tag}_{task}"

    search_pattern = os.path.join(
        base_dir,
        "derivatives",
        "fmriprep-22.0.2",
        sub_id,
        session,
        "func",
        f"{sub_id}_{session}_task-{task}*desc-confounds_timeseries.tsv",
    )
    matched_files = sorted(glob.glob(search_pattern))

    if not matched_files:
        return None, {
            f"{prefix}_confounds_found": 0,
            f"{prefix}_n_volumes": np.nan,
            f"{prefix}_mean_fd": np.nan,
            f"{prefix}_max_fd": np.nan,
            f"{prefix}_pct_fd_over_05mm": np.nan,
            f"{prefix}_mean_std_dvars": np.nan,
            f"{prefix}_qc_pass": np.nan,
        }

    conf_path = matched_files[0]
    df_conf = pd.read_csv(conf_path, sep="\t", na_values=["n/a", "NA", "null", ""])
    n_volumes = len(df_conf)

    # Scalar head-motion and signal-intensity QC metrics
    fd = pd.to_numeric(df_conf.get("framewise_displacement"), errors="coerce")
    dvars_col = "std_dvars" if "std_dvars" in df_conf.columns else "dvars"
    dvars = pd.to_numeric(df_conf.get(dvars_col), errors="coerce")

    mean_fd = float(fd.mean()) if fd.notna().any() else np.nan
    max_fd = float(fd.max()) if fd.notna().any() else np.nan
    pct_high_motion = float((fd.fillna(0.0) > fd_threshold).mean() * 100.0)
    mean_dvars = float(dvars.mean()) if dvars.notna().any() else np.nan

    # Standard fMRI scan inclusion criterion: mean FD < 0.5 mm and < 20% high-motion TRs
    qc_pass = int(pd.notna(mean_fd) and (mean_fd < fd_threshold) and (pct_high_motion < 20.0))

    qc_metrics: Dict[str, Any] = {
        f"{prefix}_confounds_found": 1,
        f"{prefix}_n_volumes": int(n_volumes),
        f"{prefix}_mean_fd": round(mean_fd, 4) if pd.notna(mean_fd) else np.nan,
        f"{prefix}_max_fd": round(max_fd, 4) if pd.notna(max_fd) else np.nan,
        f"{prefix}_pct_fd_over_05mm": round(pct_high_motion, 2),
        f"{prefix}_mean_std_dvars": round(mean_dvars, 4) if pd.notna(mean_dvars) else np.nan,
        f"{prefix}_qc_pass": qc_pass,
    }

    # Build nuisance regression matrix (Friston-24 + WM/CSF + top 5 aCompCor)
    base_motion = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    if strategy == "simple":
        candidate_cols = base_motion + ["white_matter", "csf"]
    else:
        friston24 = [
            f"{m}{suffix}"
            for m in base_motion
            for suffix in ["", "_derivative1", "_power2", "_derivative1_power2"]
        ]
        acompcor_top5 = [f"a_comp_cor_0{i}" for i in range(5)]
        candidate_cols = friston24 + ["white_matter", "csf"] + acompcor_top5

    # Include non-steady-state T1 equilibration outlier regressors if present
    nss_cols = [c for c in df_conf.columns if c.startswith("non_steady_state_outlier")]
    selected_cols = [c for c in candidate_cols + nss_cols if c in df_conf.columns]

    # Replace initial NaN in first-order motion derivatives with 0.0
    clean_confounds_df = df_conf[selected_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # Append binary spike regressors (motion scrubbing) for volumes exceeding FD threshold
    if scrubbing and fd.notna().any():
        outlier_indices = np.where(fd.fillna(0.0).values > fd_threshold)[0]
        for idx in outlier_indices:
            spike_col = np.zeros(n_volumes, dtype=float)
            spike_col[idx] = 1.0
            clean_confounds_df[f"motion_spike_tr_{idx}"] = spike_col

    return clean_confounds_df, qc_metrics




# COHORT PIPELINE & WITHIN-SUBJECT DIFFERENCE VECTORS (DELTA X = SES-02 - SES-01)
def subjects(
    patients: Iterable[str],
    tasks: Optional[Iterable[str]] = None,
    proportional_threshold: float = 0.20,
    conn_kind: str = "correlation",
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], np.ndarray]]:
    """
    Run the full multimodal feature extraction pipeline across subjects and tasks,
    returning a combined DataFrame (with ses-01, ses-02, and delta features) and
    a dictionary of 200 x 200 functional connectivity matrices.
    """
    base_dir = os.path.join(os.getcwd(), "data", "ds006110")
    task_list = list(tasks) if tasks is not None else ["rest"]

    records: List[Dict[str, Any]] = []
    fc_store: Dict[Tuple[str, str, str], np.ndarray] = {}

    # Fetch Schaefer 200 atlas once for the entire cohort
    print("Loading Schaefer 200 atlas (7 Yeo networks)...", flush=True)
    atlas_maps, ci_yeo7, net_to_id = load_schaefer_atlas(n_rois=200, yeo_networks=7)

    shared_pheno_files = [
        os.path.join(base_dir, "participants.tsv"),
        os.path.join(base_dir, "participants.json"),
        os.path.join(base_dir, "derivatives", "phenotype", "scored", "ses-01", "ASC11.tsv"),
        os.path.join(base_dir, "derivatives", "phenotype", "scored", "ses-01", "IDAS-II.tsv"),
        os.path.join(base_dir, "derivatives", "phenotype", "scored", "ses-02", "EDI.tsv"),
        os.path.join(base_dir, "derivatives", "phenotype", "scored", "ses-02", "MEQ30.tsv"),
        os.path.join(base_dir, "derivatives", "phenotype", "scored", "ses-02", "ASC11.tsv"),
    ]

    for patient in patients:
        sub_id = _normalize_subject_id(patient)
        print(f"\nProcessing {sub_id}...", flush=True)

        # 1. Demographics, dose, and psychometric scales
        patient_profile = extract_patient_clinical_demo(shared_pheno_files, subject_id=sub_id)

        # 2. Confounds (QC) -> other features
        for task in task_list:
            ses_feature_keys: Dict[str, List[str]] = {"ses01": [], "ses02": []}

            for ses in ["ses-01", "ses-02"]:
                ses_tag = ses.replace("-", "")

                print(f"  [{sub_id} | {ses} | {task}] Confounds & QC...", flush=True)
                conf_df, qc_dict = extract_confounds_and_qc(
                    base_dir=base_dir,
                    subject_id=sub_id,
                    session=ses,
                    task=task,
                    strategy="friston24_acompcor",
                    fd_threshold=0.5,
                    scrubbing=True,
                )
                patient_profile.update(qc_dict)

            # 3. Within-subject difference vector (delta X = X_ses02 - X_ses01) for regression models
            fd_s1 = patient_profile.get(f"ses01_{task}_mean_fd", np.nan)
            fd_s2 = patient_profile.get(f"ses02_{task}_mean_fd", np.nan)
            patient_profile[f"delta_{task}_mean_fd"] = (
                round(fd_s2 - fd_s1, 4) if pd.notna(fd_s1) and pd.notna(fd_s2) else np.nan
            )

            for key_s1 in ses_feature_keys["ses01"]:
                key_s2 = key_s1.replace("ses01_", "ses02_", 1)
                delta_key = key_s1.replace("ses01_", "delta_", 1)
                v1 = patient_profile.get(key_s1, np.nan)
                v2 = patient_profile.get(key_s2, np.nan)
                patient_profile[delta_key] = (
                    round(float(v2 - v1), 6) if pd.notna(v1) and pd.notna(v2) else np.nan
                )

        records.append(patient_profile)

    df_patients = pd.DataFrame(records).drop(columns=["ses02_EDI_ONE"], errors="ignore")
    return df_patients, fc_store


# MAIN EXECUTION ENTRYPOINT
if __name__ == "__main__":
    patients = ["PC001"]
    tasks = ["rest"]

    df_patients, fc_store = subjects(
        patients=patients,
        tasks=tasks,
        proportional_threshold=0.20,
        conn_kind="correlation",  # Switch to "partial correlation" for Ledoit-Wolf estimator
    )

    print("\n--- EXTRACTED FEATURES SUMMARY ---")
    print(f"Total patients: {df_patients.shape[0]}, Total extracted columns: {df_patients.shape[1]}")
    print(df_patients.T.tail(35))

    output_dir = os.path.join(os.getcwd(), "data", "processed")
    os.makedirs(output_dir, exist_ok=True)

    # Save tabular features to CSV
    csv_path = os.path.join(output_dir, "features_ses01_ses02.csv")
    df_patients.to_csv(csv_path, index=False)

    # Save FC matrices (200 x 200) to compressed NumPy .npz archive
    npz_path = os.path.join(output_dir, "fc_matrices_ses01_ses02.npz")
    np.savez_compressed(npz_path, **{f"{k[0]}_{k[1]}_{k[2]}": v for k, v in fc_store.items()})

    print(f"\nSaved tabular features to: {csv_path}")
    print(f"Saved FC matrices to:      {npz_path}")