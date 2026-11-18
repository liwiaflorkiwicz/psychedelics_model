# imports
import os
import subprocess
import matplotlib.pyplot as plt
from nilearn import image, plotting, datasets, maskers


def download_data(subject_id):
    print(f"Downloading required fMRIPrep resting-state files for {subject_id}...")

    s3_url = f"s3://openneuro.org/ds006110/derivatives/fmriprep-22.0.2/{subject_id}"
    target_dir = os.path.join(DATA_DIR, subject_id)
    os.makedirs(target_dir, exist_ok=True)

    # Pobieramy WYŁĄCZNIE: BOLD MNI, Brain Mask MNI oraz Confounds TSV
    download_cmd = (
        f"aws s3 cp {s3_url} {target_dir} --recursive "
        f"--exclude '*' "
        f"--include '*task-rest*space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz' "
        f"--include '*task-rest*space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz' "
        f"--include '*task-rest*desc-confounds_timeseries.tsv' "
        f"--no-sign-request"
    )

    result = subprocess.run(download_cmd, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        print("AWS CLI Error Output:", result.stderr)
    else:
        print("Download completed successfully.\n")
    return result

# visualization of brain 3D
def visualize_3d(bold_path, session_label):
    print(f"Wczytuję plik dla {session_label}: {bold_path}")

    if not os.path.exists(bold_path):
        print(f"PLIK NIE ISTNIEJE: {bold_path}")
        return

    print("Obliczam średni obraz BOLD (Mean BOLD)...")
    mean_bold_img = image.mean_img(bold_path)

    print("Generuję wizualizację 3D Glass Brain...")
    plotting.plot_glass_brain(
        mean_bold_img,
        title=f"{SUB_ID} | {session_label} | 3D Glass Brain Projection",
        display_mode='ortho',
        colorbar=True,
        plot_abs=False,
        threshold='auto'
    )
    plt.show()

if __name__ == "__main__":
    download_data(subject_id=SUB_ID)
    visualize_3d(file_paths["ses-01"]["bold"], "ses-01 (Baseline)")






