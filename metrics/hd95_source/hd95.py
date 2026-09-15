import numpy as np
from .metrics import compute_surface_distances, compute_robust_hausdorff


def compute_hd95(fixed, moving, moving_warped, labels):
    hd95 = []
    for i in labels:
        if ((fixed==i).sum()==0) or ((moving==i).sum()==0):
            hd95.append(np.NAN)
        else:
            hd95.append(compute_robust_hausdorff(compute_surface_distances((fixed==i), (moving_warped==i), np.ones(3)), 95.))
    mean_hd95 =  np.nanmean(hd95)
    return mean_hd95, hd95

if __name__ == '__main__':

    import SimpleITK as sitk
    import numpy as np


    fix_img = sitk.ReadImage('/root/autodl-tmp/data/OASIS/OASIS_OAS1_0438_MR1/aligned_seg35.nii.gz')
    mov_img = sitk.ReadImage('/root/autodl-tmp/data/OASIS/OASIS_OAS1_0439_MR1/aligned_seg35.nii.gz')

    fix_img = sitk.GetArrayFromImage(fix_img)
    mov_img = sitk.GetArrayFromImage(mov_img)
    
    labels = np.unique(fix_img)[1:]
    mean_hd95, _ = compute_hd95(fix_img, mov_img, mov_img, labels)
    print(mean_hd95)
