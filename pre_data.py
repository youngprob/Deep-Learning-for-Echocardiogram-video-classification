import os
import zipfile
import shutil
import pandas as pd
from tqdm import tqdm
from azure.storage.blob import BlobServiceClient
from sklearn.model_selection import train_test_split

# configuration for downloading and organizing the data
BLOB_SERVICE_URL = "https://aimistanforddatasets01.blob.core.windows.net"
SAS_TOKEN = "YOUR_SAS_TOKEN"
CONTAINER_NAME = "echonetdynamic-2"
ZIP_BLOB_NAME = "EchoNet-Dynamic.zip"

# defines the local directory structure for the data
BASE_DATA_DIR = "./data"
DOWNLOAD_DIR = os.path.join(BASE_DATA_DIR, "download")
EXTRACT_DIR = os.path.join(BASE_DATA_DIR, "extracted")
SORTED_DIR = os.path.join(BASE_DATA_DIR, "sorted")
SPLIT_DIR = os.path.join(BASE_DATA_DIR, "split") # final data folder for training

def download_data(zip_path):
    """Downloads the dataset from Azure Blob Storage."""
    os.makedirs(os.path.dirname(zip_path), exist_ok=True)
    
    if os.path.exists(zip_path):
        print(f"'{ZIP_BLOB_NAME}' already exists. Skipping download.")
        return

    print(f"Downloading '{ZIP_BLOB_NAME}'...")
    blob_service_client = BlobServiceClient(account_url=BLOB_SERVICE_URL, credential=SAS_TOKEN)
    blob_client = blob_service_client.get_blob_client(CONTAINER_NAME, ZIP_BLOB_NAME)
    blob_size = blob_client.get_blob_properties().size
    stream = blob_client.download_blob()

    with open(zip_path, "wb") as file, tqdm(
        total=blob_size, unit='B', unit_scale=True, desc=ZIP_BLOB_NAME
    ) as pbar:
        for chunk in stream.chunks():
            file.write(chunk)
            pbar.update(len(chunk))
    print("Download complete.")

def extract_data(zip_path, extract_path):
    """Extracts the dataset from the downloaded zip file."""
    expected_folder = os.path.join(extract_path, "EchoNet-Dynamic")
    if os.path.exists(expected_folder):
        print("Dataset already extracted. Skipping extraction.")
        return
        
    print(f"Extracting '{ZIP_BLOB_NAME}'...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("Extraction complete.")

def sort_videos(extract_path, sorted_path):
    """Sorts videos into 'normal' and 'abnormal' folders based on EF."""
    normal_dir = os.path.join(sorted_path, "normal_hearts")
    abnormal_dir = os.path.join(sorted_path, "abnormal_hearts")

    os.makedirs(normal_dir, exist_ok=True)
    os.makedirs(abnormal_dir, exist_ok=True)
    
    if len(os.listdir(normal_dir)) > 0 or len(os.listdir(abnormal_dir)) > 0:
        print("Videos already sorted. Skipping sorting process.")
        return

    print("Sorting videos into normal and abnormal categories...")
    base_extracted_path = os.path.join(extract_path, "EchoNet-Dynamic")
    csv_path = os.path.join(base_extracted_path, "FileList.csv")
    video_dir = os.path.join(base_extracted_path, "Videos")
    
    df = pd.read_csv(csv_path)
    
    # an EF between 50-70% is considered normal
    normal_videos = [f"{name}.avi" for name in df[(df["EF"] >= 50) & (df["EF"] <= 70)]["FileName"]]
    abnormal_videos = [f"{name}.avi" for name in df[(df["EF"] < 50) | (df["EF"] > 70)]["FileName"]]

    for video_list, dest_dir, desc in [(normal_videos, normal_dir, "Copying normal"), (abnormal_videos, abnormal_dir, "Copying abnormal")]:
        for video in tqdm(video_list, desc=desc):
            src = os.path.join(video_dir, video)
            if os.path.exists(src):
                shutil.copy(src, dest_dir)
            
    print("Sorting complete.")

def split_data(sorted_path, split_path):
    """Splits sorted videos into train, validation, and test sets."""
    train_dir = os.path.join(split_path, "train")
    if os.path.exists(train_dir) and len(os.listdir(os.path.join(train_dir, "normal_hearts"))) > 0:
        print("Data already split. Skipping split process.")
        return
        
    print("Splitting data into train, validation, and test sets...")
    val_dir = os.path.join(split_path, "val")
    test_dir = os.path.join(split_path, "test")
    
    for dir_path in [train_dir, val_dir, test_dir]:
        os.makedirs(os.path.join(dir_path, "normal_hearts"), exist_ok=True)
        os.makedirs(os.path.join(dir_path, "abnormal_hearts"), exist_ok=True)

    normal_heart_dir = os.path.join(sorted_path, "normal_hearts")
    abnormal_heart_dir = os.path.join(sorted_path, "abnormal_hearts")
        
    normal_files = os.listdir(normal_heart_dir)
    abnormal_files = os.listdir(abnormal_heart_dir)

    all_files = normal_files + abnormal_files
    labels = ['normal'] * len(normal_files) + ['abnormal'] * len(abnormal_files)

    # split into 70% train, 30% temp (val + test)
    files_train, files_temp, labels_train, labels_temp = train_test_split(
        all_files, labels, test_size=0.30, random_state=42, stratify=labels
    )
    # split temp into 50% val, 50% test (making each 15% of the total)
    files_val, files_test, labels_val, labels_test = train_test_split(
        files_temp, labels_temp, test_size=0.50, random_state=42, stratify=labels_temp
    )

    def copy_files(file_list, label_list, dest_dir):
        for file, label in tqdm(zip(file_list, label_list), desc=f"Copying to {os.path.basename(dest_dir)}"):
            src_folder = normal_heart_dir if label == 'normal' else abnormal_heart_dir
            dest_folder = os.path.join(dest_dir, "normal_hearts") if label == 'normal' else os.path.join(dest_dir, "abnormal_hearts")
            shutil.copy(os.path.join(src_folder, file), dest_folder)
            
    copy_files(files_train, labels_train, train_dir)
    copy_files(files_val, labels_val, val_dir)
    copy_files(files_test, labels_test, test_dir)
    print("Data splitting complete.")


# this block runs when the script is executed directly
if __name__ == "__main__":
    zip_file_path = os.path.join(DOWNLOAD_DIR, ZIP_BLOB_NAME)
    
    download_data(zip_file_path)
    extract_data(zip_file_path, EXTRACT_DIR)
    sort_videos(EXTRACT_DIR, SORTED_DIR)
    split_data(SORTED_DIR, SPLIT_DIR)
    

    print("\nData preparation is complete. The data is ready for training in the './data/split' directory.")
