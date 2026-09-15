import os

import numpy as np
import SimpleITK as sitk
import torch
from hydra.utils import get_original_cwd


def load_landmarks(path):
    """Load landmarks from a text file."""
    with open(path) as f:
        landmarks = np.array(
            [list(map(int, line[:-1].split("\t")[:3])) for line in f.readlines()]
        )

    return landmarks


def load_DIRLab_imgs(folder, case_idx=1, phase_idx=0, only_lung=True):
    ct_img = sitk.ReadImage(
        os.path.join(folder, "Images", f"case{case_idx}_T{phase_idx}0.nii.gz")
    )
    ct_arr = torch.FloatTensor(sitk.GetArrayFromImage(ct_img))
    params = dict(
        direction=ct_img.GetDirection(),
        origin=ct_img.GetOrigin(),
        size=ct_img.GetSize(),
        spacing=ct_img.GetSpacing(),
    )

    mask_folder = "Lungs" if only_lung else "Bodies"
    mask_img = sitk.ReadImage(
        os.path.join(folder, mask_folder, f"case{case_idx}_T{phase_idx}0.nii.gz")
    )
    mask_arr = torch.BoolTensor(sitk.GetArrayFromImage(mask_img))

    oars_img = sitk.ReadImage(
        os.path.join(folder, "OARs", f"case{case_idx}_T{phase_idx}0.nii.gz")
    )
    oars_arr = torch.FloatTensor(sitk.GetArrayFromImage(oars_img))

    return ct_arr, mask_arr, oars_arr, params


def load_DIRLab_marks(folder, case_idx=1, phase_idx=0, extremed=True):
    if extremed:
        path = os.path.join(
            folder, "ExtremePhases", f"Case{case_idx}_300_T{phase_idx}0_xyz.txt"
        )
    else:
        path = os.path.join(
            folder, "Sampled4D", f"case{case_idx}_4D-75_T{phase_idx}0.txt"
        )
    marks = load_landmarks(path)
    return torch.FloatTensor(marks)


def load_DIRLab(folder="data", case_idx=1, only_lung=True):
    folder = os.path.join(get_original_cwd(), folder, f"Case{case_idx}Pack")
    # Images
    fix_arr, fix_mask, fix_oars, params = load_DIRLab_imgs(
        folder, case_idx, 0, only_lung
    )
    mov_arr, mov_mask, mov_oars, _ = load_DIRLab_imgs(folder, case_idx, 5, only_lung)
    fix_marks = load_DIRLab_marks(folder, case_idx, 0, extremed=True)
    mov_marks = load_DIRLab_marks(folder, case_idx, 5, extremed=True)

    return dict(
        fix_arr=fix_arr,
        mov_arrs=[mov_arr],
        fix_mask=fix_mask,
        mov_mask=mov_mask,
        fix_marks=fix_marks,
        mov_marks=mov_marks,
        fix_oars=fix_oars,
        mov_oars=mov_oars,
        params=params,
    )

def load_OASIS_imgs(folder, case_idx=1):
    ct_img = sitk.ReadImage(os.path.join(folder, "OASIS_OAS1_%04d_MR1/aligned_norm.nii.gz" % case_idx))
    ct_arr = torch.FloatTensor(sitk.GetArrayFromImage(ct_img))
    params = dict(
        direction=ct_img.GetDirection(),
        origin=ct_img.GetOrigin(),
        size=ct_img.GetSize(),
        spacing=ct_img.GetSpacing(),
    )

    oars_img = sitk.ReadImage(os.path.join(folder, "OASIS_OAS1_%04d_MR1/aligned_seg35.nii.gz" % case_idx))
    oars_arr = torch.FloatTensor(sitk.GetArrayFromImage(oars_img))

    mask_img = sitk.ReadImage(os.path.join(folder, "OASIS_OAS1_%04d_MR1/aligned_ROI.nii.gz" % case_idx))
    mask_arr = torch.BoolTensor(sitk.GetArrayFromImage(mask_img))

    return ct_arr, mask_arr, oars_arr, params


def load_OASIS(folder='data', case_idx=1):
    if case_idx > 19:
        raise ValueError ("The maximum case_idx is 19!")
    folder = os.path.join(get_original_cwd(), folder)
    with open(folder + '/pairs_val.csv', 'r') as f:
        pairs = f.readlines()
    pair = pairs[case_idx].split(', ')
    fix_idx = pair[0]
    mov_idx = pair[1][:-1]
    fix_idx = int(fix_idx)
    mov_idx = int(mov_idx)

    fix_arr, fix_mask, fix_oars, params = load_OASIS_imgs(folder, fix_idx)
    mov_arr, mov_mask, mov_oars, _ = load_OASIS_imgs(folder, mov_idx)

    return dict(
        fix_arr=fix_arr,
        mov_arrs=[mov_arr],
        fix_mask=fix_mask,
        mov_mask=mov_mask,
        fix_oars=fix_oars,
        mov_oars=mov_oars,
        params=params,
    )

def load_ACDC_imgs(folder, case_idx=1, phase="D"):
    ct_img = sitk.ReadImage(os.path.join(folder, "patient%03d_E%s.nii.gz" % (case_idx, phase)))
    ct_arr = torch.FloatTensor(sitk.GetArrayFromImage(ct_img))
    params = dict(
        direction=ct_img.GetDirection(),
        origin=ct_img.GetOrigin(),
        size=ct_img.GetSize(),
        spacing=ct_img.GetSpacing(),
    )

    oars_img = sitk.ReadImage(os.path.join(folder, "patient%03d_E%s_gt.nii.gz" % (case_idx, phase)))
    oars_arr = torch.FloatTensor(sitk.GetArrayFromImage(oars_img))
    mask_arr = torch.ones_like(oars_arr).bool()

    return ct_arr, mask_arr, oars_arr, params

def load_ACDC(folder="data", case_idx=1):
    case_idx = case_idx + 100
    folder = os.path.join(get_original_cwd(), folder, "patient%03d" % case_idx)
    # Images
    fix_arr, fix_mask, fix_oars, params = load_ACDC_imgs(
        folder, case_idx, "D")
    mov_arr, mov_mask, mov_oars, _ = load_ACDC_imgs(
        folder, case_idx, "S")

    return dict(
        fix_arr=fix_arr,
        mov_arrs=[mov_arr],
        fix_mask=fix_mask,
        mov_mask=mov_mask,
        fix_oars=fix_oars,
        mov_oars=mov_oars,
        params=params,
    )
