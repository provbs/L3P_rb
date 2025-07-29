import os
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm
import nibabel as nib
import numpy as np
from pathlib import Path
import argparse

import numpy as np
import nibabel as nib

import SimpleITK as sitk
import numpy as np
from pathlib import Path

# List to track failed files
failed_files = []

def combine_masks(mask_dir):
    mask_dir = Path(mask_dir)
    masks = [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
    ]
    
    # 각 마스크 파일 경로 확인 및 존재 여부 점검
    mask_paths = {mask: mask_dir / f"{mask}.nii.gz" for mask in masks}
    existing_masks = {mask: path for mask, path in mask_paths.items() if path.exists()}

    # 파일이 하나도 없다면 에러 발생
    if not existing_masks:
        raise ValueError("No valid mask files found. Did you run TotalSegmentator successfully?")
    
    # 첫 번째 마스크 파일을 기준으로 레퍼런스 이미지 설정
    first_mask = next(iter(existing_masks.values()))
    ref_img = sitk.ReadImage(str(first_mask))
    
    # 마스크 결합을 위한 빈 배열 생성
    combined = np.zeros(sitk.GetArrayFromImage(ref_img).shape, dtype=np.uint8)
    
    # 각 마스크 파일 결합
    for mask, path in existing_masks.items():
        img = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        print(f"Combining mask: {mask}")  # 파일 로드에 대한 로그 출력
        combined[img > 0.5] = 1  # 모든 마스크를 하나의 값으로 결합
    
    # 결합된 마스크를 SimpleITK 이미지로 변환 후 반환
    combined_img = sitk.GetImageFromArray(combined)
    combined_img.CopyInformation(ref_img)  # 원본 이미지의 메타데이터 복사
    
    return combined_img


def lung_segmentation(name, input_nifti_file_path, output_nifti_file_path, merged_save_path):
    try:
        src = os.path.join(input_nifti_file_path, name)
        folder_name = name.split(".")[0]
        save_path = os.path.join(output_nifti_file_path, folder_name)

        totalsegmentator(
            src,
            save_path,
            device="gpu",
            roi_subset=[
                "lung_upper_lobe_left",
                "lung_lower_lobe_left",
                "lung_upper_lobe_right",
                "lung_middle_lobe_right",
                "lung_lower_lobe_right",
            ],
        )
        combined_img = combine_masks(save_path)
        sitk.WriteImage(combined_img, os.path.join(merged_save_path, name))
    except Exception as e:
        print(f"Error processing {name}: {e}")
        failed_files.append(name)  # Add the failed file name to the list


def process_all_folders(root_input_path, root_output_path, root_merged_path):
    for subdir, _, files in os.walk(root_input_path):
        relative_path = os.path.relpath(subdir, root_input_path)
        if relative_path == ".":
            relative_path = ""

        output_subdir = os.path.join(root_output_path, relative_path)
        merged_subdir = os.path.join(root_merged_path, relative_path)

        os.makedirs(output_subdir, exist_ok=True)
        os.makedirs(merged_subdir, exist_ok=True)

        nii_files = sorted([f for f in files if f.endswith(".nii")])
        for name in tqdm(nii_files, desc=f"Processing {relative_path}"):
            lung_segmentation(name, subdir, output_subdir, merged_subdir)

    # After processing all files, print the list of files that failed
    if failed_files:
        print("\nThe following files failed to process:")
        for failed_file in failed_files:
            print(failed_file)
    else:
        print("\nAll files processed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NIfTI files for lung segmentation")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing NIfTI files")
    parser.add_argument("--divided_save_dir", type=str, required=True, help="Directory to save divided lung segments")
    parser.add_argument("--merged_save_dir", type=str, required=True, help="Directory to save merged lung segments")

    args = parser.parse_args()

    input_nifti_root_path = args.input_dir
    divided_save_root_path = args.divided_save_dir
    merged_save_root_path = args.merged_save_dir

    process_all_folders(input_nifti_root_path, divided_save_root_path, merged_save_root_path)
