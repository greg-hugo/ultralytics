# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Convolution modules."""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = (
    "Conv",
    "Conv2",
    "LightConv",
    "DWConv",
    "DWConvTranspose2d",
    "ConvTranspose",
    "Focus",
    "GhostConv",
    "MobileOneBlock",
    "ChannelAttention",
    "SpatialAttention",
    "CBAM",
    "Concat",
    "RepConv",
    "gOctCBR"
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing."""

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """
    Light convolution with args(ch_in, ch_out, kernel).

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor."""
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        """Initialize DWConvTranspose2d class with given parameters."""
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose 2d layer."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose2d layer with batch normalization and activation function."""
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Applies transposed convolutions, batch normalization and activation to input."""
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Applies activation and convolution transpose operation to input."""
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus wh information into c-space."""

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initializes Focus object with user defined channel, convolution, padding, group and activation values."""
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """
        Applies convolution to concatenated tensor and returns the output.

        Input shape is (b,c,w,h) and output shape is (b,4c,w/2,h/2).
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initializes the GhostConv object with input channels, output channels, kernel size, stride, groups and
        activation.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Forward propagation through a Ghost Bottleneck layer with skip connection."""
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class DepthWiseConv(nn.Module):
    def __init__(self, inc, kernel_size, stride=1):
        super().__init__()
        padding = 1
        if kernel_size == 1:
            padding = 0
        # self.conv = nn.Sequential(
        #     nn.Conv2d(inc, inc, kernel_size, stride, padding, groups=inc, bias=False,),
        #     nn.BatchNorm2d(inc),
        # )
        self.conv = conv_bn(inc, inc, kernel_size, stride, padding, inc)

    def forward(self, x):
        return self.conv(x)


class PointWiseConv(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.conv = conv_bn(inc, outc, 1, 1, 0)

    def forward(self, x):
        return self.conv(x)


class SEBlock(nn.Module):
    """
    Squeeze and Excite module.

    Pytorch implementation of `Squeeze-and-Excitation Networks` -
    https://arxiv.org/pdf/1709.01507.pdf
    """

    def __init__(self, in_channels: int, rd_ratio: float = 0.0625) -> None:
        """
        Construct a Squeeze and Excite Module.

        :param in_channels: Number of input channels.
        :param rd_ratio: Input channel reduction ratio.
        """
        super(SEBlock, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels=in_channels, out_channels=int(in_channels * rd_ratio), kernel_size=1, stride=1, bias=True
        )
        self.expand = nn.Conv2d(
            in_channels=int(in_channels * rd_ratio), out_channels=in_channels, kernel_size=1, stride=1, bias=True
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply forward pass."""
        b, c, h, w = inputs.size()
        x = F.avg_pool2d(inputs, kernel_size=[h, w])
        x = self.reduce(x)
        x = F.relu(x)
        x = self.expand(x)
        x = torch.sigmoid(x)
        x = x.view(-1, c, 1, 1)
        return inputs * x


class MobileOneBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, k, stride=1, dilation=1, padding_mode="zeros", deploy=False, use_se=False
    ):
        super(MobileOneBlock, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deploy = deploy
        kernel_size = 3
        padding = 1
        assert kernel_size == 3
        assert padding == 1
        self.k = k
        padding_11 = padding - kernel_size // 2
        self.nonlinearity = nn.ReLU()

        if use_se:
            # self.se = SEBlock(out_channels, internal_neurons=out_channels // 16)
            ...
        else:
            self.se = nn.Identity()

        if deploy:
            self.dw_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=in_channels,
                bias=True,
                padding_mode=padding_mode,
            )
            self.deploy_bn = nn.BatchNorm2d(in_channels)
            self.pw_reparam = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, bias=True
            )

        else:
            self.dw_bn_layer = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            for k_idx in range(k):
                setattr(self, f"dw_3x3_{k_idx}", DepthWiseConv(in_channels, 3, stride=stride))
            self.dw_1x1 = DepthWiseConv(in_channels, 1, stride=stride)

            self.pw_bn_layer = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            for k_idx in range(k):
                setattr(self, f"pw_1x1_{k_idx}", PointWiseConv(in_channels, out_channels))

    def forward(self, inputs):
        if self.deploy:
            x = self.dw_reparam(inputs)
            x = self.deploy_bn(x)
            x = self.nonlinearity(x)
            x = self.pw_reparam(x)
            x = self.deploy_bn(x)
            x = self.nonlinearity(x)
            return x

        if self.dw_bn_layer is None:
            id_out = 0
        else:
            id_out = self.dw_bn_layer(inputs)

        x_conv_3x3 = []
        for k_idx in range(self.k):
            x = getattr(self, f"dw_3x3_{k_idx}")(inputs)
            x_conv_3x3.append(x)
        x_conv_1x1 = self.dw_1x1(inputs)

        x = id_out + x_conv_1x1 + sum(x_conv_3x3)
        x = self.nonlinearity(self.se(x))

        # 1x1 conv # https://github.com/iscyy/yoloair
        if self.pw_bn_layer is None:
            id_out = 0
        else:
            id_out = self.pw_bn_layer(x)
        x_conv_1x1 = []
        for k_idx in range(self.k):
            x_conv_1x1.append(getattr(self, f"pw_1x1_{k_idx}")(x))
        x = id_out + sum(x_conv_1x1)
        x = self.nonlinearity(x)
        return x


class RepConv(nn.Module):
    """
    RepConv is a basic rep-style block, including training and deploy status.

    This module is used in RT-DETR.
    Based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initializes Light Convolution layer with inputs, outputs & optional activation function."""
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward process."""
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward process."""
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Generates appropriate kernels and biases for convolution by fusing branches of the neural network."""
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)

# define gOctConv #
USE_BALANCE = False
up_kwargs = {'mode': 'bilinear'}


class gOctaveConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, alpha_in=[0.5, 0.5], alpha_out=[0.5, 0.5], stride=1,
                 padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs=up_kwargs):
        super(gOctaveConv, self).__init__()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.groups = groups
        self.weights = nn.Parameter(
            torch.Tensor(out_channels, round(in_channels / self.groups), kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.up_kwargs = up_kwargs
        self.h2g_pool = nn.AvgPool2d(kernel_size=(2, 2), stride=2)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.alpha_in = [0]
        tmpsum = 0
        for i in range(len(alpha_in)):
            tmpsum += alpha_in[i]
            self.alpha_in.append(tmpsum)
        self.alpha_out = [0]
        tmpsum = 0
        for i in range(len(alpha_out)):
            tmpsum += alpha_out[i]
            self.alpha_out.append(tmpsum)
        self.inbranch = len(alpha_in)
        self.outbranch = len(alpha_out)
        global USE_BALANCE
        self.use_balance = USE_BALANCE
        if self.use_balance:
            self.bals = nn.Parameter(torch.Tensor(self.outbranch, out_channels))
            init.normal_(self.bals, mean=1.0, std=0.05)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, xset):
        yset = []
        ysets = []
        for j in range(self.outbranch):
            ysets.append([])
        if isinstance(xset, torch.Tensor):
            xset = [xset, ]
        if USE_BALANCE:
            bals_norm = torch.abs(self.bals) / (torch.abs(self.bals).sum(dim=0) + 1e-14)
        for i in range(self.inbranch):
            if xset[i] is None:
                continue
            if self.stride == 2:
                x = F.avg_pool2d(xset[i], (2, 2), stride=2)
            else:
                x = xset[i]
            begin_x = int(round(self.in_channels * self.alpha_in[i] / (self.groups * self.alpha_in[-1])))
            end_x = int(round(self.in_channels * self.alpha_in[i + 1] / (self.groups * self.alpha_in[-1])))
            if begin_x == end_x:
                continue
            for j in range(self.outbranch):
                begin_y = int(round(self.out_channels * self.alpha_out[j] / self.alpha_out[-1]))
                end_y = int(round(self.out_channels * self.alpha_out[j + 1] / self.alpha_out[-1]))
                if begin_y == end_y:
                    continue
                scale_factor = 2 ** (i - j)
                # this_output_shape = xset[j].shape[2:4]
                if self.bias is not None:
                    this_bias = self.bias[begin_y:end_y]
                else:
                    this_bias = None

                if self.use_balance:
                    this_weight = self.weights[begin_y:end_y, begin_x:end_x, :, :]
                    this_weight = this_weight * bals_norm[j, begin_y:end_y].view(this_weight.shape[0], 1, 1, 1)
                else:
                    this_weight = self.weights[begin_y:end_y, begin_x:end_x, :, :]

                if scale_factor > 1:
                    y = F.conv2d(x, this_weight, this_bias, 1, self.padding, self.dilation, self.groups)
                    y = F.interpolate(y, scale_factor=scale_factor, mode=up_kwargs['mode'])
                elif scale_factor < 1:
                    x_resize = F.interpolate(x, scale_factor=scale_factor, mode=up_kwargs['mode'])
                    y = F.conv2d(x_resize, this_weight, this_bias, 1, self.padding, self.dilation, self.groups)
                else:
                    y = F.conv2d(x, this_weight, this_bias, 1, self.padding, self.dilation, self.groups)
                ysets[j].append(y)

        for j in range(self.outbranch):
            if len(ysets[j]) != 0:
                yset.append(sum(ysets[j]))
            else:
                yset.append(None)
        del ysets
        return yset


class gOctaveCBR(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, alpha_in=[0.5, 0.5], alpha_out=[0.5, 0.5],
                 stride=1, padding=1, dilation=1,
                 groups=1, bias=False, up_kwargs=up_kwargs):
        super(gOctaveCBR, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = bias
        self.std_conv = False
        if len(alpha_in) == 1 and len(alpha_out) == 1:
            self.std_conv = True
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=self.bias)
        else:
            self.conv = gOctaveConv(in_channels, out_channels, kernel_size, alpha_in, alpha_out, stride, padding,
                                    dilation, groups, bias, up_kwargs)
        self.sum = int(round(sum(alpha_out)))
        self.bns = nn.ModuleList()
        self.prelus = nn.ModuleList()
        for i in range(len(alpha_out)):
            if int(round(out_channels * alpha_out[i])) != 0:
                # groupnorm = 16 if (out_channels * alpha_out[i]) < 32 else 32
                self.bns.append(nn.BatchNorm2d(int(round(out_channels * alpha_out[i] / self.sum))))
                self.prelus.append(nn.PReLU(int(round(out_channels * alpha_out[i] / self.sum))))
            else:
                self.bns.append(None)
                self.prelus.append(None)
        self.outbranch = len(alpha_out)
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out

    def forward(self, xset):
        if self.std_conv:
            if isinstance(xset, torch.Tensor):
                xset = [xset, ]
            xset = self.conv(xset[0])
            xset = self.prelus[0](self.bns[0](xset))
        else:
            xset = self.conv(xset)
            for i in range(self.outbranch):
                if xset[i] is not None:
                    xset[i] = self.prelus[i](self.bns[i](xset[i]))
        if len(self.alpha_out) == 1:
            xset = xset[0]
        return xset
