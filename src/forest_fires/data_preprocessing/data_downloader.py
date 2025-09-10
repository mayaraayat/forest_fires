import shutil
import requests, zipfile, os
from pathlib import Path
import kagglehub


def download_forestfire_class1() -> Path:
    """Download and copy the forest fire dataset .wav files into class_1 folder."""
    
    target_dir = Path("data/forest_fire_dataset/class_1")
    target_dir.mkdir(parents=True, exist_ok=True)
    # Skip if already has wav files
    if any(target_dir.glob("*.wav")):
        return target_dir
    
    
    dataset_dir = kagglehub.dataset_download("forestprotection/forest-wild-fire-sound-dataset")
    dataset_dir = Path(dataset_dir)


    # Copy all wav files into class_1
    for wav_file in dataset_dir.rglob("*.wav"):
        shutil.copy(str(wav_file), target_dir / wav_file.name)

    return target_dir



def download_esc50_class0() -> Path:
    """Download and copy ESC-50 dataset .wav files into class_0 folder."""
    # Define target directory
    target_dir = Path("data/forest_fire_dataset/class_0")
    target_dir.mkdir(parents=True, exist_ok=True)
    # Skip if already has wav files
    if any(target_dir.glob("*.wav")):
        return target_dir
    
    # Define paths
    zip_path = 'ESC-50-master.zip'
    url = 'https://github.com/karoldvl/ESC-50/archive/master.zip'
    
    # Download the dataset
    with requests.get(url, stream=True) as r, open(zip_path, 'wb') as f:
        r.raise_for_status()
        f.write(r.content)
    
    # Extract the dataset
    extract_folder = Path('ESC-50-master')
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    
    # Copy all .wav files into class_0
    audio_folder = extract_folder / 'ESC-50-master' / 'audio'
    for wav_file in audio_folder.rglob("*.wav"):
        shutil.copy(str(wav_file), target_dir / wav_file.name)
    
    # Clean up: remove zip file and extracted folder
    zip_path_obj = Path(zip_path)
    zip_path_obj.unlink()  # remove zip
    shutil.rmtree(extract_folder)  # remove extracted folder
    
    return target_dir
