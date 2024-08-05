import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def positional_encoding(x, dim_l_embed):
    rets = []
    for i in range(dim_l_embed):
        for fn in [torch.sin, torch.cos]:
            rets.append(fn((2. ** i) * (x/1.5)))
    return torch.cat(rets, -1)

def gradient(x):
    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, 1:]
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, 1:, :]

    dx, dy = right - left, bottom - top 
    dx[:, :, -1] = 0
    dy[:, -1, :] = 0
    
    return ((dx.square().sum(dim=0) + 0.000001).sqrt() + (dy.square().sum(dim=0) + 0.000001).sqrt()) / 2.0

def output_deformed_image(deform_net, param_net, stretched_pos_test, DIM_1, DIM_2_TEST):
    with torch.no_grad():
        deformed_positions = deform_net(stretched_pos_test)
        pos = stretched_pos_test.clone()
        pos[:,1:] += deformed_positions
        deformed_positions = pos
        test_img = param_net(deformed_positions).transpose(0,1).view(3, DIM_1, DIM_2_TEST).cpu()
        return test_img
    
def show(img):
    plt.imshow(img.clone().transpose(0,1).transpose(1,2).detach().cpu())
    plt.show()    