# torch
import torch
from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn

class Transformer_2D(nn.Module):
    def __init__(self):
        super(Transformer_2D, self).__init__()
    # @staticmethod
    def forward(self,src, flow):
        b = flow.shape[0]#torch.Size([1, 2, 512, 512])
        h = flow.shape[2]
        w = flow.shape[3]
        size = (h,w)

        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b,1,1,1).cuda()
        new_locs = grid+flow#torch.Size([1, 2, 512, 512])
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1 , 0]]#torch.Size([1, 512, 512, 2])
        #提供一个input的Tensor以及一个对应的flow-field网格(比如光流，体素流等)，然后根据grid中每个位置提供的坐标信息(这里指input中pixel的坐标)，将input中对应位置的像素值填充到grid指定的位置，得到最终的输出。
        warped = F.grid_sample(src,new_locs,align_corners=True,padding_mode="border")
        # ctx.save_for_backward(src,flow)
        return warped



