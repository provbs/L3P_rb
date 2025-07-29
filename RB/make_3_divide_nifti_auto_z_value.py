import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom, gaussian_filter
from tqdm import tqdm
import math
import argparse


'''
lung mask를 기반으로 resize를 통해 middle, internal mask 생성
'''

def resize_mask(img, zoom_factor,**kwargs):
    try:
        h, w, k = img.shape[:3]

        # make zoom tuple (너비와 높이 이후의 나머지 차원에 대해서는 1을 가진 튜플로서 줌 팩터를 생성)
        zoom_tuple = (zoom_factor,) * 3 + (1,) * (img.ndim - 3)
        
        # Zoom out
        print("zoom factor :", zoom_factor)
        
        # zoom_ratio가 0.7보다 작은 경우
        if zoom_factor < 0.7:
            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            zk = int(np.round(k * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
            head = (k - zk) // 2 
            
            # Zero-padding
            out = np.zeros_like(img)
            
            # 기존 shape와 resize된 shape 크기 출력
            print("\norigin shape : ", out.shape)
            print("zoomed shape : ", zoom(img, zoom_tuple, **kwargs).shape)
            
            # 기존 크기의 shape에 resize된 mask 삽입
            out[top : top + zh, left : left + zw, head : head + zk] = zoom(
                img, zoom_tuple, **kwargs
            )
            
            # resize된 mask에서 mask가 존재하는 z slice만을 추출
            z_slices_with_label = []
            for z in range(out.shape[2]):
                slice_data = out[:, :, z]
                if np.any(slice_data == 1):
                    z_slices_with_label.append(z)

            print("z slice with label : " , z_slices_with_label)
            
            # mask가 존재하는 z slice 평균 * 0.1한 값을 조정 수치로 선정
            # print(z_slices_with_label)
            
            different = np.mean(z_slices_with_label)*0.1
            print("before : ", different)

            if math.isnan(different):
                print("NaN for z slice diffrent")
                different = 0
            else:
                # mask가 존재하는 z slice 평균 * 0.1한 값을 조정 수치로 선정'
                different = int(np.mean(z_slices_with_label)*0.1)

            # z 값 조정 수치 출력
            print("after : " , different)
            
            # diffrent 만큼 z 값 조정
            out = np.roll(out, different, axis=2)

        # zoom_ratio가 0.7 ~ 1.0인 경우
        elif 0.7 <= zoom_factor < 1:
            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            zk = int(np.round(k * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
            head = (k - zk) // 2 
            
            # resize된 mask에서 mask가 존재하는 z slice만을 추출
            out = np.zeros_like(img)
            print("\norigin shape : ", out.shape)
            print("zoomed shape : ", zoom(img, zoom_tuple, **kwargs).shape)
            out[top : top + zh, left : left + zw, head : head + zk] = zoom(
                img, zoom_tuple, **kwargs
            )
            
            # resize된 mask에서 mask가 존재하는 z slice만을 추출
            z_slices_with_label = []
            for z in range(out.shape[2]):
                slice_data = out[:, :, z]
                if np.any(slice_data == 1):
                    z_slices_with_label.append(z)

            print("z slice with label : " , z_slices_with_label)

            different = np.mean(z_slices_with_label)*0.05
            print("before : ", different)

            if math.isnan(different):
                print("NaN for z slice diffrent")
                different = 0
            else:
                # mask가 존재하는 z slice 평균 * 0.05한 값을 조정 수치로 선정'
                different = int(np.mean(z_slices_with_label)*0.05)
            
            # z 값 조정 수치 출력
            print("after : " , different)
            
            # diffrent 만큼 z 값 조정
            out = np.roll(out, different, axis=2)
            
        # zoom_ratio가 1.0을 초과하는 경우
        elif zoom_factor > 1:
            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            zk = int(np.round(k / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
            head = (k - zk) // 2
            
            out = zoom(
                img[top : top + zh, left : left + zw, head : head + zk],
                zoom_tuple,
                **kwargs
            )
            
            # `out` might still be slightly larger than `img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = (out.shape[0] - h) // 2
            trim_left = (out.shape[1] - w) // 2
            trim_head = (out.shape[2] - k) // 2
            out = out[
                trim_top : trim_top + h,
                trim_left : trim_left + w,
                trim_head : trim_head + k,
            ]

        # zoom_ratio가 1.0인 경우 (resize X)
        else:
            out = img
        return out
    except Exception as e:
        print("error : " , e)

'''
Array에서 연속적이지 않은 부분을 골라내는 함수 (region간의 경계 좌표를 알아내기 위해 사용)
'''

def find_non_consecutive_numbers(arr):                 
    non_consecutive_nums = [arr[0],arr[-1]]             # 초기값 -> array의 첫번째와 마지막 value
    for i in range(len(arr) - 1):                       # array값을 하나하나 찍어보며 불연속적인 부분을 취합      
        if arr[i+1] != arr[i] + 1:
            non_consecutive_nums.append(arr[i])
            non_consecutive_nums.append(arr[i+1])
            
    non_consecutive_nums = sorted(non_consecutive_nums)   
    # 불연속적인 집합의 앞 부분 두개의 값과 마지막 두 개의 값을 추출 (정상적으로는 region 분할이 이상적으로 되었다면 불연속적인 value 갯수가 4개이어야함)     
    non_consecutive_nums = non_consecutive_nums[:2] + non_consecutive_nums[-2:]                # left region 경계값 2개, right region 경계값 2개
    
    # 만약 불연속적인 value가 2개인 경우 중복으로 더해진 값을 제거 
    # internal region의 일부 case에서 left region 또는 right region 한 부분만 존재하여 불연속적인 value가 2개일 수 있음
    if non_consecutive_nums[1] == non_consecutive_nums[-1]:
        non_consecutive_nums = non_consecutive_nums[:2]
    return sorted(non_consecutive_nums)


'''
각 영역의 길이를 계산하는 함수
'''

def calculate_distance_ratio(img):
    y_array,z_array,external_array,medial_array = [],[],[],[]
    dim_x, dim_y, dim_z = img.shape       # nifti 이미지의 사이즈
    
    # 여러 slice 중 y평면에서 mask가 존재하는 slice만 추출
    for y in range(dim_y):
        if np.any(img[:, y, :] == 3):
            y_array.append(y)
    
    # mask가 존재하는 y slice의 중심 값 => y_mean
    y_mean = int(sum(y_array)/len(y_array))
    
    # 여러 slice 중 z평면에서 mask가 존재하는 slice만 추출
    for z in range(dim_z):
        if np.any(img[:, :, z] == 1):
            z_array.append(z)
    
    # mask가 존재하는 z slice의 중심 값 => z_mean     
    z_mean = int((sum(z_array)/len(z_array))*1.1)
    
    # [y_mean,z_mean]평면에서 external region이 존재하는 x 값 추출
    for x in range(dim_x):
        if np.any(img[x, y_mean, z_mean] == 1):
            external_array.append(x)
    
    # [y_mean,z_mean]평면에서 medial region이 존재하는 x 값 추출
    for x in range(dim_x):
        if np.any(img[x, y_mean, z_mean] == 2):
            medial_array.append(x)
    
    # region 경계값 추출   
    boundary_external_array = find_non_consecutive_numbers(external_array)        
    boundary_medial_array = find_non_consecutive_numbers(medial_array)
    
    # region 거리 추출
    external_distance_1 = boundary_external_array[1] - boundary_external_array[0]
    external_distance_2 = boundary_external_array[3] - boundary_external_array[2]
    
    medial_distance_1 = boundary_medial_array[1] - boundary_medial_array[0]

    if len(boundary_medial_array) == 2 :
        medial_distance_2 = boundary_medial_array[1] - boundary_medial_array[0]
    else:
        medial_distance_2 = boundary_medial_array[3] - boundary_medial_array[2]
    
    # external_middle ratio 계산
    external_medial_ratio = (external_distance_1 + external_distance_2) / (medial_distance_1 + medial_distance_2)
    
    return external_medial_ratio

'''
이미지를 resize하고 계산된 external_middle ratio에 따라 resize ratio를 재조정하는 자동화 함수
'''

def mask_auto_adjust_resize(
    name,
    lung_name,
    input_nifti_path,
    lung_nifti_path,
    output_nifti_path,
    zoom_factor,
    zoom_factor_2,
    **kwargs
):
    # NIfTI 파일을 읽어옵니다.
    try:
        src = os.path.join(input_nifti_path, name)
        img = nib.load(src)
        data = img.get_fdata()

        lung_src = os.path.join(lung_nifti_path, lung_name)
        lung_img = nib.load(lung_src)
        lung_data = lung_img.get_fdata()
        
        max_iteration = 0
        while True:  # 무한 반복문을 통해 조건이 충족될 때까지 반복
            # Zoom 함수를 적용하기 위해 이미지 데이터를 클립된 줌 함수에 전달합니다.
            print("ITER : ", max_iteration)
            zoomed_data = resize_mask(lung_data, zoom_factor,**kwargs)
            zoomed_data = gaussian_filter(zoomed_data, sigma=0.8)
            
            zoomed_data_2 = resize_mask(lung_data, zoom_factor_2,**kwargs)
            zoomed_data_2 = gaussian_filter(zoomed_data_2, sigma=0.8)


            zoomed_data[zoomed_data > 0.1] = 2
            zoomed_data[zoomed_data <= 0.1] = 0


            zoomed_data_2[zoomed_data_2 > 0.1] = 1
            zoomed_data_2[zoomed_data_2 <= 0.1] = 0


            merged_zoom_data = zoomed_data + zoomed_data_2
            merged_zoom_data[merged_zoom_data >= 2] = 2


            print("1!")
            
            # region label이 lung 영역내에서만 존재하도록 만듬
            combination_with_lung_mask = lung_data + merged_zoom_data
            combination_with_lung_mask = combination_with_lung_mask * lung_data
            final_data = combination_with_lung_mask.astype(np.uint8)


            print("2!")

            
            if max_iteration == 7:
                print("Maximum number of iterations reached.")
                print(calculate_distance_ratio(final_data))
                zoomed_img = nib.Nifti1Image(final_data, img.affine, img.header)
                nib.save(zoomed_img, output_nifti_path + "/" + name)
                break

            print("3!")

            if 0.95 <= calculate_distance_ratio(final_data) <= 1.05:
                print("4!")
                # 조건을 충족하면 반복문 종료
                print(calculate_distance_ratio(final_data))
                zoomed_img = nib.Nifti1Image(final_data, img.affine, img.header)
                nib.save(zoomed_img, output_nifti_path + "/" + name)
                break

            print("5!")

            if calculate_distance_ratio(final_data) < 0.95:
                zoom_factor += 0.01
            
            if calculate_distance_ratio(final_data) > 1.05:
                zoom_factor -= 0.01
            
            if max_iteration >= 5 and calculate_distance_ratio(final_data) < 0.95:
                zoom_factor += 0.005
                
            elif max_iteration >= 5 and calculate_distance_ratio(final_data) > 1.05:
                zoom_factor -= 0.005
                
            max_iteration += 1
    except Exception as e:
        print("Def mask auto error : " , e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize masks based on lung segmentation")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory containing lung + organ NIfTI files")
    parser.add_argument("--lung_dir", type=str, required=True, help="Lung mask only NIfTI directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory to save the resized NIfTI files")
    
    args = parser.parse_args()

    input_nifti_path = args.input_dir
    lung_nifti_path = args.lung_dir
    save_nifti_file_path = args.output_dir

    # 기본 줌 팩터
    zoom_factor = 0.6
    zoom_factor_2 = 0.8

    # save file path가 존재하지 않을 시 새로 만들도록 함
    if not os.path.exists(save_nifti_file_path):
        os.mkdir(save_nifti_file_path)
        print("New folder created.")
    else:
        print("Folder already exists.")


    # input_nifti_path 파일 리스트
    only_nii_list = sorted(
        [f for f in os.listdir(input_nifti_path) if (f.endswith(".gz") or f.endswith(".nii"))]
    )
        
    # lung_nifti_path에서 원하는 파일 필터링
    only_lung_nii_list = sorted(
        [f for f in os.listdir(lung_nifti_path) if (f.endswith(".gz") or f.endswith(".nii"))]
    )

    # # 디버깅용, 원하는 파일 이름 리스트
    # desired_filenames_input = ['S0106_3d.nii', 'S0107_3d.nii', 'S0192_3d.nii', 'S0251_3d.nii', 'S0368_3d.nii', 
    #                            'S0376_3d.nii', 'S0493_3d.nii', 'S0547_3d.nii', 'S0616_3d.nii', 'S0792_3d.nii', 
    #                            'S0945_3d.nii', 'S0998_3d.nii']

    # # input_nifti_path에서 원하는 파일 필터링
    # only_nii_list = sorted(
    #     [f for f in os.listdir(input_nifti_path) if (f.endswith(".gz") or f.endswith(".nii")) and f in desired_filenames_input]
    # )
    
    # # lung_nifti_path에서 원하는 파일 필터링
    # only_lung_nii_list = sorted(
    #     [f for f in os.listdir(lung_nifti_path) if (f.endswith(".gz") or f.endswith(".nii")) and f in desired_filenames_input]
    # )

    # 중심을 유지하면서 mask를 축소
    for name, lung_name in zip(tqdm(only_nii_list), only_lung_nii_list):
        mask_auto_adjust_resize(
            name,
            lung_name,
            input_nifti_path,
            lung_nifti_path,
            save_nifti_file_path,
            zoom_factor=zoom_factor,
            zoom_factor_2=zoom_factor_2
        )