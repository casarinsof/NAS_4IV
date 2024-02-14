import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.Linear_super import LinearSuper
from model.module.layernorm_super import LayerNormSuper
from model.module.multihead_super import AttentionSuper
from model.module.embedding_super import PatchembedSuper, SparseTubesTokenizerSuper
from model.utils import trunc_normal_
from model.utils import DropPath
from tubevit.positional_encoding import get_3d_sincos_pos_embed, get_2d_sincos_pos_embed
from model.module.selfattentionpooling_super import SelfAttentionPooling_Super
import numpy as np

def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

#todo continua a uniare autoformer con model di tubevit

# todo - missing: gesione dei num patches -- serve?
# todo- missing: capire cosa fa con input immagine e non volume


class Vision_TransformerSuper(nn.Module):

    def __init__(self, img_shape=None, video_shape=None, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., pre_norm=True, scale=False, gp=False, relative_position=False, change_qkv=False,
                 abs_pos = False, # I changed to False because there is the pos ebeddin of tubes now
                 max_relative_position=14):
        super(Vision_TransformerSuper, self).__init__()
        # the configs of super arch
        self.super_embed_dim = embed_dim
        # self.super_embed_dim = args.embed_dim
        self.super_mlp_ratio = mlp_ratio
        self.super_layer_num = depth
        self.super_num_heads = num_heads
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
        self.num_classes = num_classes # there is an order: first video, then img
        self.pre_norm=pre_norm
        self.scale=scale
        self.video_shape = np.array(video_shape[1:]) if video_shape is not None else None
        self.image_shape = np.array(img_shape[1:]) if img_shape is not None else None
        self.kernel_sizes = ( #todo sicuro daranno errore questi per le imamgini
            (8, 8, 8),
            (16, 4, 4),
            (4, 12, 12),
            (1, 16, 16),
        )

        self.strides = (
            (16, 32, 32),
            (6, 32, 32),
            (16, 32, 32),
            (32, 16, 16),
        )

        self.offsets = (
            (0, 0, 0),
            (4, 8, 8),
            (0, 16, 16),
            (0, 0, 0),
        )

        self.sparse_tubes_tokenizer = SparseTubesTokenizerSuper(embed_dim, self.kernel_sizes, self.strides, self.offsets)
      #  self.patch_embed_super = PatchembedSuper(img_size=img_size, patch_size=patch_size,
                                              #   in_chans=in_chans, embed_dim=embed_dim)
        self.gp = gp

        # configs for the sampled subTransformer
        # Defines configurations for the current sampled architecture. These configurations will be set using the
        # set_sample_config method.
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_layer_num = None
        self.sample_num_heads = None
        self.sample_dropout = None
        self.sample_output_dim = None

        self.blocks = nn.ModuleList()
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        for i in range(depth):
            self.blocks.append(TransformerEncoderLayer(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                                       qkv_bias=qkv_bias, qk_scale=qk_scale, dropout=drop_rate,
                                                       attn_drop=attn_drop_rate, drop_path=dpr[i],
                                                       pre_norm=pre_norm, scale=self.scale,
                                                       change_qkv=change_qkv, relative_position=relative_position,
                                                       max_relative_position=max_relative_position))

        # parameters for vision transformer
       # num_patches = self.patch_embed_super.num_patches #todo it is missing now from tubes
      #  num_patches = self.sparse_tubes_tokenizer.num_pathes

      #  self.abs_pos = abs_pos
      #  if self.abs_pos:
        #    self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        #    trunc_normal_(self.pos_embed, std=.02)
        if video_shape is not None: #only video train
            self.pos_embedding_vid = self._generate_position_embedding(self.video_shape)
            self.pos_embedding_vid = torch.nn.Parameter(self.pos_embedding_vid, requires_grad=False)
            self.register_parameter("pos_embedding_vid", self.pos_embedding_vid)
        if img_shape is not None: #only image train
            self.pos_embedding_img = self._generate_position_embedding(self.image_shape)
            self.pos_embedding_img = torch.nn.Parameter(self.pos_embedding_img, requires_grad=False)
            self.register_parameter("pos_embedding_img", self.pos_embedding_img)
        # if both Not None, both are initialized


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        trunc_normal_(self.cls_token, std=.02)
        self.register_parameter("class_token", self.cls_token)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=embed_dim)

        self.attention_pooling = SelfAttentionPooling_Super(embed_dim)

        # classifier head
        if video_shape is not None:
            self.head_vid = LinearSuper(embed_dim, num_classes[0]) if num_classes[0] > 0 else nn.Identity()
            if img_shape is not None:
                self.head_img = LinearSuper(embed_dim, num_classes[1]) if num_classes[1] > 0 else nn.Identity()
            else:
                self.head_img = LinearSuper(embed_dim, num_classes[1]) if num_classes[1] > 0 else nn.Identity()
        else:
            self.head_img = LinearSuper(embed_dim, num_classes[1]) if num_classes[1] > 0 else nn.Identity()
            self.head_vid = LinearSuper(embed_dim, num_classes[0]) if num_classes[0] > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'rel_pos_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head_vid = LinearSuper(self.embed_dim, num_classes[0]) if num_classes[0] > 0 else nn.Identity()
        self.head_img = LinearSuper(self.embed_dim, num_classes[1]) if num_classes[1] > 0 else nn.Identity()

    def set_sample_config(self, config: dict):
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_num_heads = config['num_heads']
        self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[0], self.super_embed_dim)
        #self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])
        self.sparse_tubes_tokenizer.set_sample_config(self.sample_embed_dim[0])
        self.sample_output_dim = [out_dim for out_dim in self.sample_embed_dim[1:]] + [self.sample_embed_dim[-1]]
        for i, blocks in enumerate(self.blocks):
            # not exceed sample layer number
            if i < self.sample_layer_num:
                sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                sample_attn_dropout = calc_dropout(self.super_attn_dropout, self.sample_embed_dim[i], self.super_embed_dim)
                blocks.set_sample_config(is_identity_layer=False,
                                        sample_embed_dim=self.sample_embed_dim[i],
                                        sample_mlp_ratio=self.sample_mlp_ratio[i],
                                        sample_num_heads=self.sample_num_heads[i],
                                        sample_dropout=sample_dropout,
                                        sample_out_dim=self.sample_output_dim[i],
                                        sample_attn_dropout=sample_attn_dropout)
            # exceeds sample layer number
            else:
                blocks.set_sample_config(is_identity_layer=True)
        if self.pre_norm:
            self.norm.set_sample_config(self.sample_embed_dim[-1])
        self.attention_pooling.set_sample_config(self.sample_embed_dim[-1], 1)
        if not isinstance(self.head_vid, nn.Identity) and not isinstance(self.head_img, nn.Identity): #todo scritto male
            self.head_vid.set_sample_config(self.sample_embed_dim[-1], self.num_classes[0])
            self.head_img.set_sample_config(self.sample_embed_dim[-1], self.num_classes[1])
        elif not isinstance(self.head_vid, nn.Identity) and isinstance(self.head_img, nn.Identity):
            self.head_vid.set_sample_config(self.sample_embed_dim[-1], self.num_classes[0])
        elif isinstance(self.head_vid, nn.Identity) and not isinstance(self.head_img, nn.Identity):
            self.head_img.set_sample_config(self.sample_embed_dim[-1], self.num_classes[1])
    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                if name.split('.')[0] == 'blocks' and int(name.split('.')[1]) >= config['layer_num']:
                    continue
                numels.append(module.calc_sampled_param_num())

        return sum(numels) + self.sample_embed_dim[0]* (2 +self.sparse_tubes_tokenizer.num_patches)
    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.sparse_tubes_tokenizer.get_complexity(sequence_length)
        total_flops += np.prod(self.pos_embed[..., :self.sample_embed_dim[0]].size()) / 2.0
        for blk in self.blocks:
            total_flops +=  blk.get_complexity(sequence_length+1)
        total_flops += self.head.get_complexity(sequence_length+1)
        return total_flops
    def forward_features(self, x, task):
        B, C, T, _, _ = x.shape
        if task == 1:
            assert T == 1 # double check
            self.pos_embedding = self.pos_embedding_img
        else:
            assert T != 1
            self.pos_embedding = self.pos_embedding_vid
        #x = self.patch_embed_super(x)
        x = self.sparse_tubes_tokenizer(x) #todo check

        cls_tokens = self.cls_token[..., :self.sample_embed_dim[0]].expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embedding[..., :self.sample_embed_dim[0]]

        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        # start_time = time.time()
        for blk in self.blocks:
            x = blk(x)
        # print(time.time()-start_time)
        if self.pre_norm:
            x = self.norm(x)

        if self.gp: #x 4, 197, 216
            return torch.mean(x[:, 1:] , dim=1) #x[:, 1:].shape = torch.Size([4, 196, 216])
        #todo else: quello con il self attention pooling
        else:
            x = self.attention_pooling(x)

        return x# #x[:, 0] non necessario se ho il self.attention_pooling?

    def forward(self, x, task):
        # flag activates the correct fc. if 0 -- Kinetics, if 1 -- IMNET
        x = self.forward_features(x, task)
        if task == 0:
            x = self.head_vid(x)
        else:
            x= self.head_img(x)
        return x
    def _calc_conv_shape(self, kernel_size, stride, offset, input_shape) -> np.ndarray:
        kernel_size = np.array(kernel_size)
        stride = np.array(stride)
        offset = np.array(offset)
        output = np.ceil((input_shape[[1, 2, 3]] - offset - kernel_size + 1) / stride).astype(int)
        return output

    def _generate_position_embedding(self, input_shape) -> torch.nn.Parameter:


        position_embedding = [torch.zeros(1, self.super_embed_dim)] #[(1, 450)]

        if input_shape[1] == 1:
            tube_shape = self._calc_conv_shape(self.kernel_sizes[-1], self.strides[-1], self.offsets[-1], input_shape) #only 2D
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=self.super_embed_dim,
                tube_shape=tube_shape,
                kernel_size=self.kernel_sizes[-1],
                stride=self.strides[-1],
                offset=self.offsets[-1],
            )
            position_embedding.append(pos_embed)

        else:
            for i in range(len(self.kernel_sizes)):
                tube_shape = self._calc_conv_shape(self.kernel_sizes[i], self.strides[i], self.offsets[i], input_shape)
                pos_embed = get_3d_sincos_pos_embed(
                    embed_dim=self.super_embed_dim,
                    tube_shape=tube_shape,
                    kernel_size=self.kernel_sizes[i],
                    stride=self.strides[i],
                    offset=self.offsets[i],
                )
                position_embedding.append(pos_embed)

        position_embedding = torch.cat(position_embedding, dim=0).contiguous()
        return position_embedding


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    Args:
        args (argparse.Namespace): parsed command-line arguments which
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, dropout=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, pre_norm=True, scale=False,
                 relative_position=False, change_qkv=False, max_relative_position=14):
        super().__init__()

        # the configs of super arch of the encoder, three dimension [embed_dim, mlp_ratio, and num_heads]
        self.super_embed_dim = dim
        self.super_mlp_ratio = mlp_ratio
        self.super_ffn_embed_dim_this_layer = int(mlp_ratio * dim)
        self.super_num_heads = num_heads
        self.normalize_before = pre_norm
        self.super_dropout = attn_drop
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.scale = scale
        self.relative_position = relative_position
        # self.super_activation_dropout = getattr(args, 'activation_dropout', 0)

        # the configs of current sampled arch
        self.sample_embed_dim = None
        self.sample_mlp_ratio = None
        self.sample_ffn_embed_dim_this_layer = None
        self.sample_num_heads_this_layer = None
        self.sample_scale = None
        self.sample_dropout = None
        self.sample_attn_dropout = None

        self.is_identity_layer = None
        # Attention calls QKV super -- inside there we have the sapling of the weights and biases!

        self.attn = AttentionSuper(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop,
            proj_drop=dropout, scale=self.scale, relative_position=self.relative_position, change_qkv=change_qkv,
            max_relative_position=max_relative_position
        )

        self.attn_layer_norm = LayerNormSuper(self.super_embed_dim) # -- inside layer norm super we have the sampling of the weights
        self.ffn_layer_norm = LayerNormSuper(self.super_embed_dim)
        # self.dropout = dropout
        self.activation_fn = gelu
        # self.normalize_before = args.encoder_normalize_before

        self.fc1 = LinearSuper(super_in_dim=self.super_embed_dim, super_out_dim=self.super_ffn_embed_dim_this_layer)
        self.fc2 = LinearSuper(super_in_dim=self.super_ffn_embed_dim_this_layer, super_out_dim=self.super_embed_dim)


    def set_sample_config(self, is_identity_layer, sample_embed_dim=None, sample_mlp_ratio=None, sample_num_heads=None, sample_dropout=None, sample_attn_dropout=None, sample_out_dim=None):

        if is_identity_layer:
            self.is_identity_layer = True
            return

        self.is_identity_layer = False

        self.sample_embed_dim = sample_embed_dim
        self.sample_out_dim = sample_out_dim
        self.sample_mlp_ratio = sample_mlp_ratio
        self.sample_ffn_embed_dim_this_layer = int(sample_embed_dim*sample_mlp_ratio)
        self.sample_num_heads_this_layer = sample_num_heads

        self.sample_dropout = sample_dropout
        self.sample_attn_dropout = sample_attn_dropout
        self.attn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)

        self.attn.set_sample_config(sample_q_embed_dim=self.sample_num_heads_this_layer*64, sample_num_heads=self.sample_num_heads_this_layer, sample_in_embed_dim=self.sample_embed_dim)

        self.fc1.set_sample_config(sample_in_dim=self.sample_embed_dim, sample_out_dim=self.sample_ffn_embed_dim_this_layer)
        self.fc2.set_sample_config(sample_in_dim=self.sample_ffn_embed_dim_this_layer, sample_out_dim=self.sample_out_dim)

        self.ffn_layer_norm.set_sample_config(sample_embed_dim=self.sample_embed_dim)


    def forward(self, x):
        """
        Args:
            x (Tensor): input to the layer of shape `(batch, patch_num , sample_embed_dim)`

        Returns:
            encoded output of shape `(batch, patch_num, sample_embed_dim)`. See fig.3 right of paper.
        """
        if self.is_identity_layer:
            return x

        # compute attn
        # start_time = time.time()

        residual = x
        # Apply layer normalization to the input tensor before the attention operation (before=True). Its a function
        # defined below
        x = self.maybe_layer_norm(self.attn_layer_norm, x, before=True)
        x = self.attn(x)
        x = F.dropout(x, p=self.sample_attn_dropout, training=self.training)
        x = self.drop_path(x)
        x = residual + x
        x = self.maybe_layer_norm(self.attn_layer_norm, x, after=True)
        # print("attn :", time.time() - start_time)


        # compute the ffn
        # start_time = time.time()
        # compute feed forward (scheme paper after grey linear fig3)
        # Compute Feedforward Network (FFN):

        # Store the output of the attention operation as the residual for later use.
        # Apply layer normalization to the input tensor before the feedforward network (before=True).

        residual = x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, before=True)
        # Apply the activation function (GELU) to the output of the first linear layer (self.activation_fn(self.fc1(x))).
        # Apply dropout to the output of the activation function using the specified dropout rate (self.sample_dropout).
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        # Pass the result through the second linear layer (self.fc2(x)).
        # Apply dropout to the output of the second linear layer using the specified dropout rate (self.sample_dropout).
        x = self.fc2(x)
        x = F.dropout(x, p=self.sample_dropout, training=self.training)
        # If scaling is enabled (self.scale), scale the output by a factor based on the specified scaling ratio.
        # Apply drop path regularization.
        if self.scale:
            x = x * (self.super_mlp_ratio / self.sample_mlp_ratio)
        x = self.drop_path(x)
        # Add the residual tensor to the output of the feedforward network.
        # Apply layer normalization to the combined tensor after the feedforward network (after=True).
        x = residual + x
        x = self.maybe_layer_norm(self.ffn_layer_norm, x, after=True)
        # print("ffn :", time.time() - start_time)
        return x

    def maybe_layer_norm(self, layer_norm, x, before=False, after=False):
        assert before ^ after
        if after ^ self.normalize_before:
            return layer_norm(x)
        else:
            return x
    def get_complexity(self, sequence_length):
        total_flops = 0
        if self.is_identity_layer:
            return total_flops
        total_flops += self.attn_layer_norm.get_complexity(sequence_length+1)
        total_flops += self.attn.get_complexity(sequence_length+1)
        total_flops += self.ffn_layer_norm.get_complexity(sequence_length+1)
        total_flops += self.fc1.get_complexity(sequence_length+1)
        total_flops += self.fc2.get_complexity(sequence_length+1)
        return total_flops

def calc_dropout(dropout, sample_embed_dim, super_embed_dim):
    return dropout * 1.0 * sample_embed_dim / super_embed_dim





