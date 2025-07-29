import os
from totalsegmentator.python_api import totalsegmentator
from scipy.ndimage import binary_closing, binary_fill_holes, binary_opening
import subprocess
from tqdm import tqdm
import nibabel as nib
import numpy as np
from pathlib import Path
import argparse

def dilation_with_fill_hole(data):
    mask_data = data
    closed_mask_data = binary_closing(mask_data.astype(np.uint8), iterations=20)
    filled_mask_data = binary_fill_holes(closed_mask_data.astype(int))
    opened_mask_data = binary_opening(filled_mask_data, iterations=10)
    filled_mask_data = binary_fill_holes(opened_mask_data.astype(int))
    return filled_mask_data

def combine_lung_organ_masks(mask_dir):
    mask_dir = Path(mask_dir)
    lung_organ_masks = [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
        "heart",
        "aorta",
        "superior_vena_cava",
        "esophagus",
        "pulmonary_vein",
        "trachea",
        "brachiocephalic_trunk",
        "brachiocephalic_vein_left",
        "brachiocephalic_vein_right",
        "atrial_appendage_left"
    ]

    ref_img = None
    for mask in lung_organ_masks:
        if (mask_dir / f"{mask}.nii.gz").exists():
            ref_img = nib.load(mask_dir / f"{lung_organ_masks[0]}.nii.gz")
        else:
            raise ValueError(f"Could not find {mask_dir / mask}.nii.gz. Did you run TotalSegmentator successfully?")

    combined = np.zeros(ref_img.shape, dtype=np.uint8)
    for idx, mask in enumerate(lung_organ_masks):
        if (mask_dir / f"{mask}.nii.gz").exists():
            img = nib.load(mask_dir / f"{mask}.nii.gz").get_fdata()
            combined[img > 0.5] = 1
            
    combined = dilation_with_fill_hole(combined)
    return nib.Nifti1Image(combined, ref_img.affine, ref_img.header)

def combine_lung_masks(mask_dir):
    mask_dir = Path(mask_dir)
    lung_organ_masks = [
        "lung_upper_lobe_left",
        "lung_lower_lobe_left",
        "lung_upper_lobe_right",
        "lung_middle_lobe_right",
        "lung_lower_lobe_right",
    ]

    ref_img = None
    for mask in lung_organ_masks:
        if (mask_dir / f"{mask}.nii.gz").exists():
            ref_img = nib.load(mask_dir / f"{lung_organ_masks[0]}.nii.gz")
        else:
            raise ValueError(f"Could not find {mask_dir / mask}.nii.gz. Did you run TotalSegmentator successfully?")

    combined = np.zeros(ref_img.shape, dtype=np.uint8)
    for idx, mask in enumerate(lung_organ_masks):
        if (mask_dir / f"{mask}.nii.gz").exists():
            img = nib.load(mask_dir / f"{mask}.nii.gz").get_fdata()
            combined[img > 0.5] = 1
            
    combined = dilation_with_fill_hole(combined)
    return nib.Nifti1Image(combined, ref_img.affine, ref_img.header)

def lung_organ_segmentation(name, input_nifti_file_path, output_nifti_file_path, merged_save_path, lung_save_path):
    src = os.path.join(input_nifti_file_path, name)
    folder_name = name.split(".")[0]
    save_path = os.path.join(output_nifti_file_path, folder_name)
    print(src, save_path)
    
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
            "heart",
            "aorta",
            "superior_vena_cava",
            "esophagus",
            "pulmonary_vein",
            "trachea",
            "brachiocephalic_trunk",
            "brachiocephalic_vein_left",
            "brachiocephalic_vein_right",
            "atrial_appendage_left"
        ],
    )
    
    aorta_src = os.path.join(save_path, "aorta.nii.gz")
    aorta = nib.load(aorta_src)
    aorta_data = aorta.get_fdata()
    aorta_data[:, :, :100] = 0
    processed_aorta = nib.Nifti1Image(aorta_data, aorta.affine, aorta.header)
    nib.save(processed_aorta, aorta_src)
    
    combined_lung_organ_mask = combine_lung_organ_masks(save_path)
    lung_mask = combine_lung_masks(save_path)
    
    nib.save(combined_lung_organ_mask, os.path.join(merged_save_path, name))
    nib.save(lung_mask, os.path.join(lung_save_path, name))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process NIfTI files for lung and organ segmentation")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing NIfTI files")
    parser.add_argument("--lung_input_dir", type=str, required=True, help="Directory to save lung masks")
    parser.add_argument("--output_divided_save_dir", type=str, required=True, help="Directory to save lung and organ masks")
    parser.add_argument("--output_merged_save_dir", type=str, required=True, help="Directory to save merged masks")

    args = parser.parse_args()

    input_nifti_file_path = args.input_dir
    lung_save_path = args.lung_input_dir
    divided_save_nifti_file_path = args.output_divided_save_dir
    merged_save_path = args.output_merged_save_dir

    os.makedirs(divided_save_nifti_file_path, exist_ok=True)
    os.makedirs(merged_save_path, exist_ok=True)
    os.makedirs(lung_save_path, exist_ok=True)

    only_nii_list = sorted([f for f in os.listdir(input_nifti_file_path) if f.endswith(".gz") or f.endswith(".nii")])

    for name in tqdm(only_nii_list):
        try:
            lung_organ_segmentation(
                name, input_nifti_file_path, divided_save_nifti_file_path, merged_save_path, lung_save_path
            )
        except Exception as e:
            print(f"Error processing {name}: {e}")
