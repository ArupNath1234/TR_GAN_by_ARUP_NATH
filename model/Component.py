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

class MLP(nn.Module):
    """Multi layer perception.
     parameter:
     in_features:int 
     hidden_features: int
     out_features: int
     p: dropout probability
    
     Attribute:
     fc:nn.Linear  first linear layer
     act:nn.GELU  GELU activation function
     fc2: nn.linear second linear layer
     drop: nn.Dropout
     Dropout layer

      """

    def __init__(self,in_features,hidden_features,out_features,p=0.):
        super().__init__()
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act=nn.GELU()
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop=nn.Dropout()
    
    def forward(self,x):
        """
        input:
        (n_samples,patches+1,in_features)

        output:
        (n_samples,patches+1,out_features)
        
        """
        x=self.fc1(x) #(n_samples,patches+1,hidden_features)
        x=self.act(x) #(n_samples,patches+1,hidden_features)
        x=self.drop(x) #(n_samples,patches+1,hidden_features)
        x=self.fc2(x) #(n_samples,patches+1,out_features)
        x=self.drop(x) #(n_samples,patches+1,out_features)

        return x
        
class Attention(nn.Module):
    """Attention mechanism
    parameter:
    dim: int  (input  and output dimension of per token feature)
    n_heads : int (Number of attention heads)
    qkv_bias: bool (if true we include bias to query, key and value projections)
    attn_p:float (Dropout probability applied to the query, key and value tensors)
    proj_p: float (Dropout probability applied to the output tensor)


    attributes:
    scale:float (normalized constant for the dot product )
    qkv:nn.linear  (linear projection for the query,key and value.)
    proj:nn.linear (linear mapping that takes in concatenated output of all attention heads and maps into a new space)
    attn_drop,proj_drop: nn.Dropout layers

    """
    def __init__(self,dim,n_heads=6,qkv_bias=True,attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads=n_heads
        self.dim=dim
        self.head_dim=dim//n_heads
        self.scale=self.head_dim **-0.5
        self.qkv=nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop=nn.Dropout(attn_p)
        self.proj=nn.Linear(dim,dim)
        self.proj_p=nn.Dropout(proj_p)


    def forward(self,x):
        """
        input x: torch tensor(shape (n_samples,n_patches+1,dim))

        output: torch tensor(shape(n_samples,n_patches+1,dim))
        """

        n_samples,n_tokens, dim=x.shape
        if dim!=self.dim:
            raise ValueError
        qkv=self.qkv(x) # (n_samples,n_patches+1,3*dim) 
        qkv=qkv.reshape(n_samples,n_tokens,3,self.n_heads,self.head_dim) #(n_samples,n_patches+1,3,n_heads,head_dim)
        qkv=qkv.permute(2,0,3,1,4) #(3,n_samples,n_heads,n_patches+1,head_dim)

        q,k,v=qkv[0],qkv[1],qkv[2]
        k_t=k.transpose(-2,-1)  #(n_samples,n_heads,head_dim,n_patches+1)
        
        dp=(q@k_t)*self.scale #(n_samples,n_heads,patches+1,patches+1)

        attn=dp.softmax(dim=-1) #(n_samples,n_heads,patches+1,patches+1)
        attn=self.attn_drop(attn)

        weighted_avg=attn@v #(n_samples,n_heads,patches+1,head_dim)

        weighted_avg= weighted_avg.transpose(1,2) #(n_samples,patches+1,n_heads,head_dim)

        weighted_avg=weighted_avg.flatten(2) #(n_samples,patches+1,dim)

        x=self.proj(weighted_avg)  #(n_samples,patches+1,dim)

        x=self.proj_p(x)  #(n_samples,patches+1,dim)

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
    """ Transformer Block
    Parameter:

    dim: int 
     embedded dimension

    n_heads: int 
      number of attention heads
    
    mlp_ratio: float
    determines the hidden dimension size of MLP module with respect to dim.

    qkv_bias: bool
    If true then we include bias to the query, key and value projections.

    p,attn_p: float

    Dropout probability

    


    attribute:

    norm1,norm2: LayerNorm
    layer normalization

    attn: Attention Attention module

    mlp:MLP MLP module.
    """

    def __init__(self,dim,n_heads,mlp_ratio=4.0,qkv_bias=True,p=0,attn_p=0.0):
        super().__init__()
        self.norm1=nn.LayerNorm(dim,eps=1e-6)
        self.attn=Attention(dim,n_heads=n_heads,qkv_bias=qkv_bias,attn_p=attn_p,proj_p=p)

        self.norm2=nn.LayerNorm(dim,eps=1e-6)

        hidden_features= int(dim*mlp_ratio)

        self.mlp=MLP(in_features=dim,hidden_features=hidden_features,out_features=dim)
    

    def forward(self,x):
        """
        input x:(n_samples,n_patches+1,dim)
        output x:(n_samples,n_patches+1,dim)
        """


        x=x+self.attn(self.norm1(x))
        y=self.mlp(self.norm2(x))
        x=x+y

        return x

class PatchEmbed(nn.Module):
    """split image into patches and embed them
    Parameters:
    img_size:int(2dimension)
    patch_size:int(2dimension)
    in_chans:int(number of input channels: for RGB its 3)
    embed_dim: int (the embeding dimension)
    
    Attributes:
    n_patches:int number of patches(h*w/n^2)
    proj: nn.Conv2d
     splitting into patches and embedding
    """

    def __init__(self,img_size=96,patch_size=6,in_chans=3,embed_dim=108):
        super().__init__()
        self.img_size=img_size
        self.patch_size=patch_size
        self.n_patches=(img_size//patch_size)**2

        self.proj=nn.Conv2d(in_chans,embed_dim,kernel_size=patch_size,stride=patch_size)

    def forward(self,x):
        """
        parameter:
        x:torch tensor(n_samples,embed_dim,n_patches**0.5,n_patches**0.5)

        return:
        x:torch tensor(n_samples,n_patches,embed_dim)

        """
        x=self.proj(x) #(n_samples,embed_dim,n_patches**0.5,n_patches**0.5)
        x=x.flatten(2) #(n_samples,embed_dim,n_patches)
        x=x.transpose(1,2) #(n_samples,n_patches,embed_dim)

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
    def __init__(self, N_p=2, N_z=50):
        super(Decoder, self).__init__()
        Fconv_layers = [
            Fconv_unit(320, 160),                   #Bx160x6x6
            Fconv_unit(160, 256),                   #Bx256x6x6
            Fconv_unit(256, 256, unsampling=True),  #Bx256x12x12
            Fconv_unit(256, 128),                   #Bx128x12x12
            Fconv_unit(128, 192),                   #Bx192x12x12
            Fconv_unit(192, 192, unsampling=True),  #Bx192x24x24
            Fconv_unit(192, 96),                    #Bx96x24x24
            Fconv_unit(96, 128),                    #Bx128x24x24
            Fconv_unit(128, 128, unsampling=True),  #Bx128x48x48
            Fconv_unit(128, 64),                    #Bx64x48x48
            Fconv_unit(64, 64),                     #Bx64x48x48
            Fconv_unit(64, 64, unsampling=True),    #Bx64x96x96
            Fconv_unit(64, 32),                     #Bx32x96x96
            Fconv_unit(32, 3)                       #Bx3x96x96
        ]

        self.Fconv_layers = nn.Sequential(*Fconv_layers)
        self.fc = nn.Linear(320+N_p+N_z, 320*6*6)

    def forward(self, input):
        x = self.fc(input)
        x = x.view(-1, 320, 6, 6)
        x = self.Fconv_layers(x)
        return x

class Decoder_transformer(nn.Module):
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
        self.fc = nn.Linear(320+N_p+N_z, 320*6*6)
        self.embed1 = nn.Embedding(320,3*96)
       
       
        # stochastic depth decay rule
        
        self.decoder =  nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=3*96, nhead= num_heads, dropout=0.1), \
                                             num_layers=patch_size, norm=nn.LayerNorm(normalized_shape=3*96, eps=1e-6))
        
        # Classifier head
        self.head = nn.Linear(3*96, 3*96*96) if num_classes > 0 else nn.Identity()


    def forward(self, x, src_mask=None, tgt_key_padding_mask=None):
            print(x.shape)
            x = self.fc(x)
            t= torch.tensor(x).to(torch.int64)
            print(t.shape)
            embedded=self.embed1(t)
            print(embedded.shape)
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
    def __init__(self, img_size=96, patch_size=6, in_chans=3, n_classes=320, embed_dim=108, depth=6,
                 n_heads=6, mlp_ratio=4., qkv_bias=True, p=0.,attn_p=0.):
        super().__init__()
        self.patch_embed=PatchEmbed(img_size=img_size,patch_size=patch_size,in_chans=in_chans,embed_dim=embed_dim)
        self.class_token=nn.Parameter(torch.zeros(1,1,embed_dim)) # declare the class token
        self.pos_embed= nn.Parameter(torch.zeros(1,1+self.patch_embed.n_patches,embed_dim))

        self.pos_drop=nn.Dropout(p=p)

        self.blocks=nn.ModuleList([
            Block(
                dim=embed_dim,
                n_heads=n_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                p=p,
                attn_p=attn_p
            )
            for _ in range(depth)
        ])

        self.norm=nn.LayerNorm(embed_dim,eps=1e-6)
        self.head=nn.Linear(embed_dim,n_classes)


    def forward(self,x):
        """
        input:(n_samples,in_chans,img_size,img_size)

        output: (n_samples,n_classes)
        """
        n_samples=x.shape[0]
        x=self.patch_embed(x)
        class_token=self.class_token.expand(n_samples,-1,-1)  #(n_samples,1,embed_dim)
        x=torch.cat((class_token,x), dim=1)  #(n_samples,1+patches,embed_dim)
        x=x + self.pos_embed #(n_samples,1+patches,embed_dim)

        x=self.pos_drop(x)

        for block in self.blocks:
            x=block(x)

        x=self.norm(x)
        class_token_final=x[:,0]
        x=self.head(class_token_final)

        return x

class Encoder_transformer(nn.Module):
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
    def __init__(self,N_p=2, N_z=50, img_size=96, patch_size=6, in_chans=3, num_classes=318, embed_dim=124, depth=6,
                    num_heads=4, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., 
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
        
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=0.1), \
                                             num_layers=patch_size, norm=nn.LayerNorm(normalized_shape=embed_dim, eps=1e-6))
        
        # Classifier head
        self.head = nn.Linear(embed_dim, 96*96) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        return x

    def forward(self, x, src_mask=None, src_key_padding_mask=None):
            #print(x.shape)
            #x = self.forward_features(x)
            #print(x.shape)
            x=x.view(3,3,124)
            x = self.encoder(x, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
            #print(x.shape)
            output = self.head(x)
            output=output.view(3,3,96,96)
            #print(output.shape)
            return output  

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
        self.dec = Decoder_transformer(N_p, N_z)

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
