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


# 4. BOLD TIME SERIES, TR, AND FUNCTIONAL CONNECTIVITY (PEARSON-Z / LEDOIT-WOLF)
def extract_bold_and_fc(
    base_dir: str,
    subject_id: str,
    session: str,
    task: str,
    atlas_maps: str,
    clean_confounds: Optional[pd.DataFrame] = None,
    conn_kind: str = "correlation",  # "correlation" (Pearson + Fisher-z) or "partial correlation" (Ledoit-Wolf)
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """
    Read TR from the NIfTI header, parcellate preprocessed 4D BOLD images using
    the Schaefer 200 atlas with confound denoising (0.01-0.1 Hz bandpass), and
    compute the 200 x 200 functional connectivity (FC) matrix.
    """
    sub_id = _normalize_subject_id(subject_id)
    func_dir = os.path.join(base_dir, "derivatives", "fmriprep-22.0.2", sub_id, session, "func")

    bold_matches = sorted(
        glob.glob(os.path.join(func_dir, f"{sub_id}_{session}_task-{task}*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"))
    )
    mask_matches = sorted(
        glob.glob(os.path.join(func_dir, f"{sub_id}_{session}_task-{task}*space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"))
    )

    if not bold_matches or not mask_matches:
        return None, None, np.nan

    bold_path = bold_matches[0]
    mask_path = mask_matches[0]

    # 1. Read exact repetition time (TR) from NIfTI header
    bold_img = nib.load(bold_path)
    tr = float(bold_img.header.get_zooms()[3])

    # 2. Parcellate and bandpass-filter BOLD signal
    masker = maskers.NiftiLabelsMasker(
        labels_img=atlas_maps,
        mask_img=mask_path,
        standardize="zscore_sample",
        detrend=True,
        low_pass=0.1,
        high_pass=0.01,
        t_r=tr,
        memory="nilearn_cache",
        verbose=0,
    )
    time_series = masker.fit_transform(bold_path, confounds=clean_confounds)

    # 3. Compute functional connectivity matrix (Pearson or Ledoit-Wolf partial correlation)
    conn_measure = ConnectivityMeasure(kind=conn_kind, standardize="zscore_sample")
    fc_raw = conn_measure.fit_transform([time_series])[0]
    np.fill_diagonal(fc_raw, 0.0)

    if conn_kind == "correlation":
        # Apply Fisher r-to-z transformation (clipped to avoid +/-inf)
        fc_clipped = np.clip(fc_raw, -0.999999, 0.999999)
        fc_matrix = np.arctanh(fc_clipped)
    else:
        # Retain regularized partial correlation matrix from Ledoit-Wolf estimator
        fc_matrix = fc_raw

    np.fill_diagonal(fc_matrix, 0.0)
    return time_series, fc_matrix, tr


# 5. GLOBAL GRAPH METRICS & MESOSCOPIC / NODAL METRICS (7 YEO NETWORKS)
def extract_graph_topology_metrics(
    fc_matrix: Optional[np.ndarray],
    ci_yeo: np.ndarray,
    net_to_id: Dict[str, int],
    session: str,
    task: str,
    proportional_threshold: float = 0.20,
    n_null_models: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute whole-brain global graph metrics and subnetwork-aggregated nodal metrics:
      A) Global metrics:
         - mean_pos_fc, modularity_Q, efficiency (E_glob), char_path_len (L),
           clustering_coef (C), small_worldness (sigma), system_segregation,
           rich_club_norm_hub (phi_norm), assortativity.
      B) Mesoscopic & nodal metrics (averaged per 7 Yeo subnetworks):
         - part_coef_{net}: mean participation coefficient,
         - within_fc_{net}: mean within-network functional connectivity,
         - strength_{net}: mean node strength (s_i),
         - betweenness_{net}: mean normalized betweenness centrality (BC_i),
         - nodal_eff_{net}: mean nodal efficiency (E_nodal, i).
    """
    ses_tag = session.replace("-", "")
    glob_prefix = f"{ses_tag}_{task}_glob"
    net_prefix = f"{ses_tag}_{task}_net"

    sorted_nets = sorted(net_to_id.items(), key=lambda x: x[1])

    # Pre-populate dictionary with NaNs in case of missing scan
    metrics: Dict[str, Any] = {
        f"{glob_prefix}_mean_pos_fc": np.nan,
        f"{glob_prefix}_modularity_Q": np.nan,
        f"{glob_prefix}_efficiency": np.nan,
        f"{glob_prefix}_char_path_len": np.nan,
        f"{glob_prefix}_clustering_coef": np.nan,
        f"{glob_prefix}_small_worldness": np.nan,
        f"{glob_prefix}_system_segregation": np.nan,
        f"{glob_prefix}_rich_club_norm_hub": np.nan,
        f"{glob_prefix}_assortativity": np.nan,
    }
    for net_name, _ in sorted_nets:
        for m_type in ["part_coef", "within_fc", "strength", "betweenness", "nodal_eff"]:
            metrics[f"{net_prefix}_{m_type}_{net_name}"] = np.nan

    if fc_matrix is None:
        return metrics

    rng = np.random.default_rng(seed)
    n_nodes = fc_matrix.shape[0]

    # 1. Keep positive weights only and zero out self-connections
    w_pos = np.where(fc_matrix > 0, fc_matrix, 0.0)
    np.fill_diagonal(w_pos, 0.0)

    triu_mask = np.triu(np.ones((n_nodes, n_nodes), dtype=bool), k=1)
    pos_edges = w_pos[triu_mask]
    pos_edges = pos_edges[pos_edges > 0]
    mean_pos_fc = float(np.mean(pos_edges)) if pos_edges.size > 0 else 0.0

    # 2. System Segregation index (Chan et al., 2014): (W_within - W_between) / W_within
    same_net_mask = (ci_yeo[:, None] == ci_yeo[None, :]) & triu_mask
    diff_net_mask = (ci_yeo[:, None] != ci_yeo[None, :]) & triu_mask
    w_within_all = float(np.mean(w_pos[same_net_mask])) if np.any(same_net_mask) else 0.0
    w_between_all = float(np.mean(w_pos[diff_net_mask])) if np.any(diff_net_mask) else 0.0
    system_segregation = (
        (w_within_all - w_between_all) / w_within_all if w_within_all > 0 else np.nan
    )

    # 3. Proportional thresholding (fixed density 15-25%) and weight normalization to [0, 1]
    w_thresh = bct.threshold_proportional(w_pos, p=proportional_threshold, copy=True)
    w_norm = bct.weight_conversion(w_thresh, "normalize", copy=True)

    # 4. Modularity (Q) via Louvain community detection
    if hasattr(bct, "community_louvain"):
        _, modularity_q = bct.community_louvain(w_norm, seed=int(rng.integers(0, 10_000)))
    else:
        _, modularity_q = bct.modularity_louvain_und(w_norm, seed=int(rng.integers(0, 10_000)))

    # 5. Global efficiency (E_glob)
    global_eff = float(bct.efficiency_wei(w_norm))

    # 6. Shortest-path distance matrix and characteristic path length (L)
    length_mat = bct.weight_conversion(w_norm, "lengths", copy=True)
    dist_mat, _ = bct.distance_wei(length_mat)
    char_path_len, _, _, _, _ = bct.charpath(dist_mat, include_diagonal=False, include_infinite=False)

    # 7. Mean clustering coefficient (C)
    clustering_vec = bct.clustering_coef_wu(w_norm)
    mean_clustering = float(np.mean(clustering_vec))

    # 8. Empirical Rich-Club (phi(k)) and Maslov-Sneppen null models (Small-Worldness + normalized Rich-Club)
    rc_emp = np.asarray(bct.rich_club_wu(w_norm), dtype=float)

    c_rand_list: List[float] = []
    l_rand_list: List[float] = []
    rc_rand_list: List[np.ndarray] = []

    for _ in range(n_null_models):
        w_rand, _ = bct.randmio_und(w_norm, itr=2)
        c_rand_list.append(float(np.mean(bct.clustering_coef_wu(w_rand))))

        l_mat_rand = bct.weight_conversion(w_rand, "lengths", copy=True)
        d_mat_rand, _ = bct.distance_wei(l_mat_rand)
        l_r, _, _, _, _ = bct.charpath(d_mat_rand, include_diagonal=False, include_infinite=False)
        l_rand_list.append(float(l_r))

        rc_r = np.asarray(bct.rich_club_wu(w_rand), dtype=float)
        rc_rand_list.append(rc_r)

    mean_c_rand = float(np.mean(c_rand_list))
    mean_l_rand = float(np.mean(l_rand_list))

    if mean_c_rand > 0 and mean_l_rand > 0 and char_path_len > 0:
        small_worldness = (mean_clustering / mean_c_rand) / (float(char_path_len) / mean_l_rand)
    else:
        small_worldness = np.nan

    # Normalized Rich-Club coefficient averaged across the high-degree hub regime (75th-90th percentile)
    degrees = np.sum(w_norm > 0, axis=1)
    k_low = max(1, int(np.percentile(degrees, 75)))
    k_high = max(k_low + 1, int(np.percentile(degrees, 90)))

    rich_club_norm_hub = np.nan
    if rc_rand_list:
        max_len = min(rc_emp.size, min(arr.size for arr in rc_rand_list))
        k_end = min(k_high, max_len)

        if k_end > k_low:
            emp_slice = rc_emp[k_low:k_end]
            rand_stack = np.vstack([arr[k_low:k_end] for arr in rc_rand_list])

            valid_counts = np.sum(np.isfinite(rand_stack), axis=0)
            rand_slice = np.divide(
                np.nansum(rand_stack, axis=0),
                valid_counts,
                out=np.full(emp_slice.shape, np.nan, dtype=float),
                where=valid_counts > 0,
            )

            valid_rc = np.isfinite(emp_slice) & np.isfinite(rand_slice) & (rand_slice > 0)
            if np.any(valid_rc):
                rich_club_norm_hub = float(np.mean(emp_slice[valid_rc] / rand_slice[valid_rc]))

    # 9. Weighted assortativity
    assortativity = float(bct.assortativity_wei(w_norm, flag=0))

    metrics.update({
        f"{glob_prefix}_mean_pos_fc": round(mean_pos_fc, 5),
        f"{glob_prefix}_modularity_Q": round(float(modularity_q), 5),
        f"{glob_prefix}_efficiency": round(global_eff, 5),
        f"{glob_prefix}_char_path_len": round(float(char_path_len), 5),
        f"{glob_prefix}_clustering_coef": round(mean_clustering, 5),
        f"{glob_prefix}_small_worldness": round(float(small_worldness), 5) if pd.notna(small_worldness) else np.nan,
        f"{glob_prefix}_system_segregation": round(float(system_segregation), 5) if pd.notna(system_segregation) else np.nan,
        f"{glob_prefix}_rich_club_norm_hub": round(float(rich_club_norm_hub), 5) if pd.notna(rich_club_norm_hub) else np.nan,
        f"{glob_prefix}_assortativity": round(assortativity, 5),
    })

    # B. Nodal metric vectors -> aggregated per 7 Yeo subnetworks
    part_coef_vec = bct.participation_coef(w_norm, ci_yeo)
    strength_vec = bct.strengths_und(w_norm)
    # Normalized betweenness centrality: BC_i / ((N - 1)(N - 2))
    betweenness_vec = bct.betweenness_wei(length_mat) / float((n_nodes - 1) * (n_nodes - 2))
    # Nodal efficiency: E_nodal,i = (1 / (N - 1)) * sum_{j != i} (1 / d_ij)
    inv_dist = np.divide(
        1.0,
        dist_mat,
        out=np.zeros_like(dist_mat, dtype=float),
        where=(dist_mat > 0) & np.isfinite(dist_mat),
    )
    nodal_eff_vec = np.sum(inv_dist, axis=1) / float(n_nodes - 1)

    for net_name, net_id in sorted_nets:
        mask = ci_yeo == net_id
        sub_fc = w_pos[np.ix_(mask, mask)]
        sub_triu = sub_fc[np.triu_indices(sub_fc.shape[0], k=1)]

        metrics[f"{net_prefix}_part_coef_{net_name}"] = round(float(np.nanmean(part_coef_vec[mask])), 5)
        metrics[f"{net_prefix}_within_fc_{net_name}"] = round(float(np.mean(sub_triu)),
                                                              5) if sub_triu.size > 0 else np.nan
        metrics[f"{net_prefix}_strength_{net_name}"] = round(float(np.mean(strength_vec[mask])), 5)
        metrics[f"{net_prefix}_betweenness_{net_name}"] = round(float(np.mean(betweenness_vec[mask])), 6)
        metrics[f"{net_prefix}_nodal_eff_{net_name}"] = round(float(np.mean(nodal_eff_vec[mask])), 5)

    return metrics

#################

# COHORT PIPELINE & WITHIN-SUBJECT DIFFERENCE VECTORS (DELTA X = SES-02 - SES-01)
# COHORT PIPELINE & WITHIN-SUBJECT DIFFERENCE VECTORS (DELTA X = SES-02 - SES-01)
def subjects(
    patients: Iterable[str],
    tasks: Optional[Iterable[str]] = None,
    proportional_threshold: float = 0.20,
    conn_kind: str = "correlation",
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], np.ndarray]]:
    """
    Run the full multimodal feature extraction pipeline across subjects and tasks,
    saving per-subject full feature tables to data/processed/<sub-PCXXX>/
    and returning a combined DataFrame plus a dictionary of FC matrices.
    """
    base_dir = os.path.join(os.getcwd(), "data", "ds006110")
    processed_base_dir = os.path.join(os.getcwd(), "data", "processed")
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

        # Create per-subject output directory: data/processed/sub-PCXXX/
        subject_out_dir = os.path.join(processed_base_dir, sub_id)
        os.makedirs(subject_out_dir, exist_ok=True)

        # 1. Demographics, dose, and psychometric scales
        patient_profile = extract_patient_clinical_demo(shared_pheno_files, subject_id=sub_id)

        # Save per-subject questionnaire & demographic results
        df_sub_questionnaires = pd.DataFrame([patient_profile]).drop(columns=["ses02_EDI_ONE"], errors="ignore")
        q_csv_path = os.path.join(subject_out_dir, f"{sub_id}_questionnaires.csv")
        df_sub_questionnaires.to_csv(q_csv_path, index=False)
        print(f"  [{sub_id}] Saved questionnaires to: {q_csv_path}", flush=True)

        # 2. Confounds (QC) -> BOLD & FC -> Graph topology metrics
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

                print(f"  [{sub_id} | {ses} | {task}] BOLD time series & FC ({conn_kind})...", flush=True)
                ts_bold, fc_mat, tr = extract_bold_and_fc(
                    base_dir=base_dir,
                    subject_id=sub_id,
                    session=ses,
                    task=task,
                    atlas_maps=atlas_maps,
                    clean_confounds=conf_df,
                    conn_kind=conn_kind,
                )
                patient_profile[f"{ses_tag}_{task}_tr"] = round(tr, 4) if pd.notna(tr) else np.nan

                if fc_mat is not None:
                    fc_store[(sub_id, ses, task)] = fc_mat
                    triu_idx = np.triu_indices(fc_mat.shape[0], k=1)
                    upper_vals = fc_mat[triu_idx]
                    pos_vals = upper_vals[upper_vals > 0]

                    fc_summary = {
                        f"{ses_tag}_{task}_fc_mean": round(float(np.mean(upper_vals)), 5),
                        f"{ses_tag}_{task}_fc_std": round(float(np.std(upper_vals)), 5),
                        f"{ses_tag}_{task}_fc_pos_mean": round(float(np.mean(pos_vals)), 5) if pos_vals.size > 0 else np.nan,
                    }
                    patient_profile.update(fc_summary)
                    ses_feature_keys[ses_tag].extend(fc_summary.keys())
                else:
                    empty_fc_summary = {
                        f"{ses_tag}_{task}_fc_mean": np.nan,
                        f"{ses_tag}_{task}_fc_std": np.nan,
                        f"{ses_tag}_{task}_fc_pos_mean": np.nan,
                    }
                    patient_profile.update(empty_fc_summary)
                    ses_feature_keys[ses_tag].extend(empty_fc_summary.keys())

                # Wywołanie ekstrakcji 44 metryk topologicznych (globalnych + 7 sieci Yeo)
                print(f"  [{sub_id} | {ses} | {task}] Global & 7-network graph metrics...", flush=True)
                graph_metrics = extract_graph_topology_metrics(
                    fc_matrix=fc_mat,
                    ci_yeo=ci_yeo7,
                    net_to_id=net_to_id,
                    session=ses,
                    task=task,
                    proportional_threshold=proportional_threshold,
                    n_null_models=10,
                    seed=42,
                )
                patient_profile.update(graph_metrics)
                ses_feature_keys[ses_tag].extend(graph_metrics.keys())

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

        # Zapis kompletnego profilu (wszystkich cech) pojedynczego pacjenta do jego folderu
        df_sub_full = pd.DataFrame([patient_profile]).drop(columns=["ses02_EDI_ONE"], errors="ignore")
        full_sub_csv_path = os.path.join(subject_out_dir, f"{sub_id}_all_features.csv")
        df_sub_full.to_csv(full_sub_csv_path, index=False)
        print(f"  [{sub_id}] Saved ALL features ({df_sub_full.shape[1]} cols) to: {full_sub_csv_path}", flush=True)

        records.append(patient_profile)

    df_patients = pd.DataFrame(records).drop(columns=["ses02_EDI_ONE"], errors="ignore")
    return df_patients, fc_store


# MAIN EXECUTION ENTRYPOINT
if __name__ == "__main__":
    patients = ["PC201"]           # initiate patients
    tasks = ["rest"]               # and tasks for analysis

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
    all_subjects_dir = os.path.join(output_dir, "all_subjects")
    os.makedirs(all_subjects_dir, exist_ok=True)

    # 1. Save / append cohort-level tabular features to CSV without duplicates
    csv_path = os.path.join(all_subjects_dir, "features_ses01_ses02.csv")
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df_combined = pd.concat([df_existing, df_patients], ignore_index=True)
        df_combined = df_combined.drop_duplicates(subset=["participant_id"], keep="last")
        df_combined = df_combined.sort_values("participant_id").reset_index(drop=True)
    else:
        df_combined = df_patients.sort_values("participant_id").reset_index(drop=True)

    df_combined.to_csv(csv_path, index=False)

    # 2. Save / append FC matrices (200 x 200) to compressed NumPy .npz archive
    npz_path = os.path.join(all_subjects_dir, "fc_matrices_ses01_ses02.npz")
    merged_fc: Dict[str, np.ndarray] = {}
    if os.path.exists(npz_path):
        with np.load(npz_path) as existing_npz:
            for key in existing_npz.files:
                merged_fc[key] = existing_npz[key]

    for (sub_id, ses, task_name), matrix in fc_store.items():
        merged_fc[f"{sub_id}_{ses}_{task_name}"] = matrix

    if merged_fc:
        np.savez_compressed(npz_path, **merged_fc)

    print(f"\nSaved cohort tabular features ({len(df_combined)} subjects, {df_combined.shape[1]} cols) to: {csv_path}")
    print(f"Saved FC matrices ({len(merged_fc)} matrices) to: {npz_path}")