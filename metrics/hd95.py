from .hd95_source.hd95 import compute_hd95

def comp_hd95(fix_seg, mov_seg, warp_seg, labels):
    fix_seg = fix_seg.cpu().detach().numpy()
    mov_seg = mov_seg.cpu().detach().numpy()
    warp_seg = warp_seg.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()

    hd95, _ = compute_hd95(fix_seg, mov_seg, warp_seg, labels)
    return hd95
