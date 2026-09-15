import torch
import torch.nn as nn
import SimpleITK as sitk


def jacobian_determinant(disp):
    device = disp.device

    gradz = nn.Conv3d(3, 3, (3, 1, 1), padding=(1, 0, 0), bias=False, groups=3)
    gradz.weight.data[:, 0, :, 0, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradz.to(device)
    grady = nn.Conv3d(3, 3, (1, 3, 1), padding=(0, 1, 0), bias=False, groups=3)
    grady.weight.data[:, 0, 0, :, 0] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    grady.to(device)
    gradx = nn.Conv3d(3, 3, (1, 1, 3), padding=(0, 0, 1), bias=False, groups=3)
    gradx.weight.data[:, 0, 0, 0, :] = torch.tensor([-0.5, 0, 0.5]).view(1, 3).repeat(3, 1)
    gradx.to(device)

    jacobian = torch.cat((gradz(disp), grady(disp), gradx(disp)), 0) + torch.eye(3, 3, device=device).view(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]
    jacdet = jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) - \
                jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) + \
                jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] - jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :])
    
    return jacdet

def comp_folding_rate(disp, out_jacobian_mask=False):
    jacobian_det = jacobian_determinant(disp)

    if out_jacobian_mask:
        # get jacobian determination 
        padding = torch.nn.ReplicationPad3d(2)
        jacobian_arr = jacobian_det.unsqueeze(0).unsqueeze(0)
        jacobian_mask = (padding(jacobian_arr).squeeze() >= 0).float().cpu().detach().numpy()
    else:
        jacobian_mask = None

    negative = (jacobian_det < 0).sum()
    total = jacobian_det.reshape(-1).shape[0]
    folding = negative / total
    return folding, jacobian_mask
