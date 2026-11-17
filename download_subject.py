import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from botocore.exceptions import ClientError

# project config
BUCKET_NAME = "openneuro.org"
DATASET_ID = "ds006110"
FMRIPREP_VERSION = "fmriprep-22.0.2"

# local directory
LOCAL_BASE_DIR = os.path.join(os.getcwd(), "data", DATASET_ID)

# number of subject to download
SUBJECTS = ["sub-PC002"]

# s3 init
s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

def download_s3_file(s3_key: str, local_path: str) -> None:
    """Downloads single file from S3, if it's not existing locally yet."""
    if os.path.exists(local_path):
        print(f"[skipped - already existing] {local_path}")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    try:
        print(f"[downloading] {s3_key} ...")
        s3.download_file(BUCKET_NAME, s3_key, local_path)
    except ClientError as e:
        print(f"[err] Cannot download {s3_key}: {e}")

# download questionnaires data
def download_phenotype_and_participants() -> None:
    """Dowlnloads demographic tables and results of psychometric scales (EDI, MEQ30, ASC11)."""
    files_to_fetch = [
        f"{DATASET_ID}/participants.tsv",
        f"{DATASET_ID}/participants.json",
        f"{DATASET_ID}/derivatives/phenotype/demographics.tsv",
        f"{DATASET_ID}/derivatives/phenotype/scored/ses-01/ASC11.tsv",
        f"{DATASET_ID}/derivatives/phenotype/scored/ses-01/ASC11.tsv",
        f"{DATASET_ID}/derivatives/phenotype/scored/ses-02/EDI.tsv",
        f"{DATASET_ID}/derivatives/phenotype/scored/ses-02/MEQ30.tsv",
        f"{DATASET_ID}/derivatives/phenotype/scored/ses-02/ASC11.tsv",
    ]

    print("\n--- downloading demographic and phenotypical data ---")
    for s3_key in files_to_fetch:
        rel_path = s3_key.replace(f"{DATASET_ID}/", "")
        local_path = os.path.join(LOCAL_BASE_DIR, *rel_path.split("/"))
        download_s3_file(s3_key, local_path)


# download fmri preprocessed data
# can define tasks or subjects
def download_preprocessed_fmri(subject_id: str, task: str = "task-rest") -> None:
    """Downloads clean fMRIPrep data (BOLD MNI, Brain Mask, Confounds) for given patient."""
    sub_id = subject_id if subject_id.startswith("sub-") else f"sub-{subject_id}"
    prefix = f"{DATASET_ID}/derivatives/{FMRIPREP_VERSION}/{sub_id}/"
    paginator = s3.get_paginator("list_objects_v2")

    print(f"\n-- downloading data for {sub_id} ({task}) ---")
    found_any = False

    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3_key = obj["Key"]

            # Filter by task (e.g. task-rest) and only needed files
            if task not in s3_key:
                continue

            is_bold = s3_key.endswith("space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
            is_mask = s3_key.endswith("space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz")
            is_confounds = s3_key.endswith("desc-confounds_timeseries.tsv")

            if is_bold or is_mask or is_confounds:
                found_any = True
                rel_path = s3_key.replace(f"{DATASET_ID}/", "")
                local_path = os.path.join(LOCAL_BASE_DIR, *rel_path.split("/"))
                download_s3_file(s3_key, local_path)

    if not found_any:
        print(f"[warn] Cannot find files for {sub_id} and task {task}.")


if __name__ == "__main__":
    # needed only once
    # download_phenotype_and_participants()

    for sub in SUBJECTS:
        download_preprocessed_fmri(subject_id=sub, task="task-rest")        # only rest and only sub 001 [for now]

    print(f"\nFinished. Saved files in: {LOCAL_BASE_DIR}")