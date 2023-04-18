import torch
import torch.nn as nn
import torch.nn.functional as F
from taming.modules.vqvae.quantize import GumbelQuantize
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class Attention(nn.Module):
    def __init__(self, dim, shuffle_rate, heads = 1, dim_head = 64):
        super().__init__()
        self.inner_dim = dim_head *  heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        # self.conv1_1 = nn.Conv2d(in_channels=dim, out_channels = dim // shuffle_rate,
        #                          kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.down = nn.PixelUnshuffle(shuffle_rate)
        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim * shuffle_rate**2, self.inner_dim * 3, bias = False)
        self.to_out = nn.Linear(self.inner_dim, dim * shuffle_rate**2)
        self.up = nn.PixelShuffle(shuffle_rate)

        # self.conv1_2 = nn.Conv2d(in_channels=dim // shuffle_rate, out_channels=dim,
        #                          kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, x):
        #x = self.conv1_1(x)
        x = self.down(x)

        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2).contiguous()

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        x = self.to_out(out)

        x = x.transpose(1, 2).contiguous().view(B, C, H, W)

        x = self.up(x)
        #x = self.conv1_2(x)

        return x


# Convolution Attention Module (CAM)
class CAM(nn.Module):
    def __init__(self, dim, shuffle_rate, exp_rate = 2):
        super(CAM, self).__init__()
        self.dim = dim
        self.shuffle_rate = shuffle_rate
        self.conv1_1 = nn.Conv2d(in_channels=dim, out_channels=dim,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.norm = LayerNorm2d(dim)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.dim // 2, out_channels=dim // 2 * exp_rate,
                      kernel_size=3, padding=1, stride=1, groups=1, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=dim // 2 * exp_rate, out_channels=dim // 2,
                      kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        )
        self.att = Attention(dim=dim//2, shuffle_rate=shuffle_rate)
        self.conv1_2 = nn.Conv2d(in_channels=dim, out_channels=dim,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, x):
        att_x, conv_x = torch.split(self.norm(self.conv1_1(x)), (self.dim // 2, self.dim // 2), dim=1)
        att_x = self.att(att_x)
        conv_x = self.conv(conv_x)
        x = self.conv1_2(torch.cat([att_x, conv_x], 1)) + x

        return x

#Dual-Path Feed-Forward Network (DP-FFN)
class DP_FFN(nn.Module):
    def __init__(self, dim, exp_rate = 2):
        super(DP_FFN, self).__init__()

        self.dim = dim

        self.norm = LayerNorm2d(dim)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 2 * exp_rate,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=dim // 2 * exp_rate, out_channels=dim // 2 * exp_rate,
                      kernel_size=3, padding=1, stride=1, groups=dim//2*exp_rate, bias=True),
            nn.GELU()
        )
        self.linear = nn.Sequential(
            nn.Conv2d(in_channels=dim // 2, out_channels=dim // 2 * exp_rate,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.GELU()
        )

        self.conv1_1 = nn.Conv2d(in_channels=dim*exp_rate, out_channels=dim,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=True)

    def forward(self, x):
        conv_x, linear_x = torch.split(self.norm(x), (self.dim//2, self.dim//2), dim=1)
        conv_x = self.conv(conv_x)
        linear_x = self.linear(linear_x)
        x = self.conv1_1(torch.cat([conv_x, linear_x], 1)) + x

        return x

# Multi-Head Blend Block (MHBB)
class MHBBlock(nn.Module):
    def __init__(self, dim, n_head, shuffle_rate, exp_rate):
        super(MHBBlock, self).__init__()
        self.n_head = n_head
        self.cam = [CAM(dim // n_head, shuffle_rate) for _ in range(n_head)]
        self.cam = nn.ModuleList(self.cam)
        self.conv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.dp_ffn = DP_FFN(dim , exp_rate)

    def forward(self, x):
        inputs = torch.chunk(x, self.n_head, 1)
        features = []
        for i in range(self.n_head):
            feature = self.cam[i](inputs[i])
            features.append(feature)
        x = torch.cat([i for i in features], 1)
        x = self.conv(x)
        x = self.dp_ffn(x)

        return x

# Mapping Network
class MapNet(nn.Module):
    def __init__(self, dim, nz=256):
        super(MapNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=nz,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=nz, out_channels=nz,
                      kernel_size=3, padding=1, stride=1, groups=nz, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=nz, out_channels=nz,
                      kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Conv2d(in_channels=nz, out_channels=nz,
                      kernel_size=3, padding=1, stride=1, groups=nz, bias=True),
            nn.GELU()
        )
        self.att = nn.Sequential(
            Attention(dim=nz, shuffle_rate=2),
            Attention(dim=nz, shuffle_rate=2)
        )

        self.quantize = GumbelQuantize(nz, nz,
                                       n_embed=8192,
                                       kl_weight=1.0e-08, temp_init=1.0,
                                       remap=None)

    def forward(self, x):
        x = self.conv(x)
        x = self.att(x)
        quant, emb_loss, info = self.quantize(x)

        return quant


#Restoration Network
class RAHC(nn.Module):

    def __init__(self, img_channel=3, width=64, nz=256, middle_blk_num=8, exp_rate=2, enc_blk_nums=[2, 4, 6],
                 dec_blk_nums=[2, 2, 2], head_nums=[1, 2, 4, 8], shuffle_rates=[16, 8, 4, 2]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_pre = nn.ModuleList()
        self.middle_aft = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()


        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(
                nn.Sequential(
                    *[MHBBlock(chan, head_nums[i], shuffle_rates[i], exp_rate) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_pre = \
            nn.Sequential(
                *[MHBBlock(chan, head_nums[-1], shuffle_rates[-1], exp_rate) for _ in range(middle_blk_num//2)]
            )

        self.mapping_network = MapNet(dim=chan, nz=nz)
        self.post_conv = nn.Conv2d(chan + nz, chan, 1)

        self.middle_aft = \
            nn.Sequential(
                *[MHBBlock(chan, head_nums[-1], shuffle_rates[-1], exp_rate) for _ in range(middle_blk_num // 2)]
            )

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[MHBBlock(chan, head_nums[len(dec_blk_nums)-1-i], shuffle_rates[len(dec_blk_nums)-1-i], exp_rate) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        mid_pre = self.middle_pre(x)
        f_rv = self.mapping_network(mid_pre.detach())
        x = self.middle_aft(self.post_conv(torch.cat([mid_pre, f_rv.detach()], 1)))

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W], f_rv

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


if __name__ == '__main__':
    net = RAHC().cuda()

    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)

    params = float(params[:-3])
    macs = float(macs[:-4])

    print(macs, params)


