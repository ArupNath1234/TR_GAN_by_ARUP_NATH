import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms
from itertools import repeat
import collections.abc

def Tensor2Image(img):
    """
    input (FloatTensor)
    output (PIL.Image)
    """
    img = img.cpu()
    img = img * 0.5 + 0.5
    img = transforms.ToPILImage()(img)
    return img

def one_hot(label, depth):
    """
    Return the one_hot vector of the label given the depth.
    Args:
        label (LongTensor): shape(batchsize)
        depth (int): the sum of the labels
    output: (FloatTensor): shape(batchsize x depth) the label indicates the index in the output
    >>> label = torch.LongTensor([0, 0, 1])
    >>> one_hot(label, 2)
    <BLANKLINE>
     1  0
     1  0
     0  1
    [torch.FloatTensor of size 3x2]
    <BLANKLINE>
    """
    out_tensor = torch.zeros(len(label), depth)
    for i, index in enumerate(label):
        out_tensor[i][index] = 1
    return out_tensor

def weights_init_normal(m):
    if isinstance(m, nn.ConvTranspose2d):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Conv2d):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.Linear):
        init.uniform_(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        init.uniform_(m.weight.data, 0.02, 1)
        init.constant_(m.bias.data, 0.0)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
        
class Attention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    
    def __init__(self, img_size=96, patch_size=6, in_chans=3, embed_dim=108):
        super().__init__()
        def _ntuple(n):
            def parse(x):
                if isinstance(x, collections.abc.Iterable):
                    return x
                return tuple(repeat(x, n))
            return parse
        to_2tuple = _ntuple(2)
        
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
        
class conv_unit(nn.Module):
    """The base unit used in the network.
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> net = conv_unit(3, 32)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 32, 96, 96])
    >>> net = conv_unit(3, 16, pooling=True)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 16, 48, 48])
    """

    def __init__(self, in_channels, out_channels, pooling=False):
        super(conv_unit, self).__init__()

        if pooling:
            layers = [nn.ZeroPad2d([0, 1, 0, 1]), nn.Conv2d(in_channels, out_channels, 3, 2, 0)]
        else:
            layers = [nn.Conv2d(in_channels, out_channels, 3, 1, 1)]

        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x

class Fconv_unit(nn.Module):
    """The base unit used in the network.
    >>> input = Variable(torch.randn(4, 64, 48, 48))
    >>> net = Fconv_unit(64, 32)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 32, 48, 48])
    >>> net = Fconv_unit(64, 16, unsampling=True)
    >>> output = net(input)
    >>> output.size()
    torch.Size([4, 16, 96, 96])
    """

    def __init__(self, in_channels, out_channels, unsampling=False):
        super(Fconv_unit, self).__init__()

        if unsampling:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 2, 1), nn.ZeroPad2d([0, 1, 0, 1])]
        else:
            layers = [nn.ConvTranspose2d(in_channels, out_channels, 3, 1, 1)]

        layers.extend([nn.BatchNorm2d(out_channels), nn.ELU()])

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        x = self.layers(input)
        return x

class Decoder(nn.Module):
    """
    Args:
        N_p (int): The sum of the poses
        N_z (int): The dimensions of the noise
    >>> Dec = Decoder()
    >>> input = Variable(torch.randn(4, 372))
    >>> output = Dec(input)
    >>> output.size()
    torch.Size([4, 3, 96, 96])
    """
    def __init__(self,N_p=2, N_z=50, img_size=96, patch_size=6, in_chans=3, num_classes=318, embed_dim=108, depth=6,
                    num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
                    drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.fc = nn.Linear(320+N_p+N_z, 512)
        self.embed1 = nn.Embedding(108,embed_dim)
       
       
        # stochastic depth decay rule
        
        self.decoder =  nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, nhead= num_heads, dropout=0.1), \
                                             num_layers=patch_size, norm=nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6))
        
        # Classifier head
        self.head = nn.Linear(embed_dim, 3*96*96)


    def forward(self, x, src_mask=None, tgt_key_padding_mask=None):
            #print(x.shape)
            x = self.fc(x)
            t= torch.tensor(x).to(torch.int64)
            #print(t.shape)
            embedded=self.embed1(t)
            #print(embedded.shape)
            output = self.decoder(embedded, mask=src_mask, src_key_padding_mask=None)
            class_token=output[:,0]
            output = self.head(class_token)
            output=output.view(3, 3, 96, 96)
            return output


class Encoder(nn.Module):
    """
    The single version of the Encoder.
    >>> Enc = Encoder()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = Enc(input)
    >>> output.size()
    torch.Size([4, 320])
    """
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=96, patch_size=6, in_chans=3, num_classes=320, embed_dim=108, depth=6,
                 num_heads=6, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


class Generator(nn.Module):
    """
    >>> G = Generator()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> pose = Variable(torch.randn(4, 2))
    >>> noise = Variable(torch.randn(4, 50))
    >>> output = G(input, pose, noise)
    >>> output.size()
    torch.Size([4, 3, 96, 96])
    """
    def __init__(self, N_p=2, N_z=50, single=True):
        super(Generator, self).__init__()
        self.enc = Encoder()
        self.dec = Decoder(N_p, N_z)

    def forward(self, input, pose, noise):
        x = self.enc(input)
        x = torch.cat((x, pose, noise), 1)
        x = self.dec(x)
        return x

class Discriminator(nn.Module):
    """
    Args:
        N_p (int): The sum of the poses
        N_d (int): The sum of the identities
    >>> D = Discriminator()
    >>> input = Variable(torch.randn(4, 3, 96, 96))
    >>> output = D(input)
    >>> output.size()
    torch.Size([4, 503])
    """
    def __init__(self, N_p=2, N_d=500):
        super(Discriminator, self).__init__()
        #Because Discriminator uses same architecture as that of Encoder
        self.enc = Encoder() 
        self.fc = nn.Linear(320, N_d+N_p+1)

    def forward(self,input):
        x = self.enc(input)
        x = x.view(-1, 320)
        x = self.fc(x)
        return x
