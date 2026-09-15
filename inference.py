import os

import hydra
from omegaconf import DictConfig

import registrators


@hydra.main(version_base=None, config_path="configs", config_name="gaussian.yaml")
def main(cfg: DictConfig):
    cfg.resume = True
    cfg.training = False
    cfg.save_dir = os.path.join("outputs", cfg.exp_name, 'Case%d' % cfg.dataset.case_idx)
    result_dir = os.path.join(cfg.save_dir, "results")
    os.makedirs(result_dir, exist_ok=True)

    registrator = registrators.__dict__[cfg.registrator.type](cfg)

    if hasattr(registrator.dataset, 'fix_marks'):
        registrator.val_marks()
    else:
        registrator.val_dice()
        registrator.val_hd95()
    registrator.val_mae()

    registrator.arr2nii(
        registrator.inf_flow(),
        os.path.join(result_dir, "flow.nii.gz"),
    )
    registrator.arr2nii(
        registrator.inf_warped(),
        os.path.join(result_dir, "warped_img.nii.gz"),
    )
    registrator.arr2nii(
        registrator.inf_segmentation(),
        os.path.join(result_dir, "warped_seg.nii.gz"),
    )
    registrator.val_folding_rate(out_jacobian_mask=False)


if __name__ == "__main__":
    main()
