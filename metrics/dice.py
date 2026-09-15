import torch


def comp_dice(fix_oars, warp_oars, labels):
    fix_masks = [fix_oars == i for i in labels]
    mov_masks = [warp_oars == i for i in labels]
    inter_areas = [(fix_masks[i] & mov_masks[i]).sum() for i in range(len(labels))]
    union_areas = [(fix_masks[i].sum() + mov_masks[i].sum()) for i in range(len(labels))]
    dices = [2 * inter_areas[i] / union_areas[i] for i in range(len(labels))]
    dices = torch.mean(torch.stack(dices)).cpu().item()
    return dices
