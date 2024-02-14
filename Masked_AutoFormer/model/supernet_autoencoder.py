import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.Linear_super import LinearSuper
from model.module.layernorm_super import LayerNormSuper
from model.module.multihead_super import AttentionSuper
from model.module.embedding_super import PatchembedSuper
from model.utils import trunc_normal_
from model.utils import DropPath
import model.video_vit as video_vit
import numpy as np

def gelu(x: torch.Tensor) -> torch.Tensor:
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float()).type_as(x)
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Vision_AutoEncoderSuper(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
             #    num_classes=1000, todo to re add when no autoencoder structure
                 embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., pre_norm=True, scale=False, gp=False, relative_position=False,
                 change_qkv=False, abs_pos = True, max_relative_position=14,
                 # decoding parameters:
                 decoder_embed_dim=512,
                 decoder_depth=8,
                 decoder_num_heads=16,
                 # Video parameters
                 norm_pix_loss=False,
                 num_frames=16,
                 t_patch_size=4,
                 sep_pos_embed=False, #todo cos'e'? - per i video, lo tengo
                 trunc_init=False, #todo cos'e'? -- tronca w e k, io terrei weights come in Autoformer dunque per il momento non lo metto
                 cls_embed=False, #todo cos'e'? -- anche questo sembra essere utile
                 pred_t_dim=8, #todo needed for time
                 ):
        super(Vision_AutoEncoderSuper, self).__init__()
        # the configs of super arch
        self.super_embed_dim = embed_dim
        self.trunc_init = trunc_init
        # self.super_embed_dim = args.embed_dim
        # -------------------------- taken from MAE --------------------------- #
        self.sep_pos_embed = sep_pos_embed
        self.cls_embed = cls_embed
        self.pred_t_dim = pred_t_dim
        self.t_pred_patch_size = t_patch_size * pred_t_dim // num_frames
        # -------------------------- taken from MAE --------------------------- #
        self.super_mlp_ratio = mlp_ratio
        self.super_layer_num = depth
        self.super_num_heads = num_heads
        self.super_dropout = drop_rate
        self.super_attn_dropout = attn_drop_rate
     #   self.num_classes = num_classes
        self.pre_norm=pre_norm
        self.decoder_embed_dim = decoder_embed_dim
        self.scale=scale
        self.patch_embed_super = PatchembedSuper(img_size=img_size, patch_size=patch_size,
                                                 in_chans=in_chans, embed_dim=embed_dim,
                                                 frames=num_frames,
                                                 t_patch_size=t_patch_size,
                                                 )
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
        num_patches = self.patch_embed_super.num_patches
        # -------------------------- added from MAE ------------------------------------
        input_size = self.patch_embed_super.input_size
        self.input_size = input_size # input size = (8, 14, 14)

      #  self.abs_pos = abs_pos
      #  if self.abs_pos:
       #     self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
       #     trunc_normal_(self.pos_embed, std=.02)
        if sep_pos_embed:  # todo in Tubes ViT it's sine and cosine fixed
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )  # shape 1, 196, 1024
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)  # 1, 8, 1024
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )
        # -------------------------- modifided from MAE ------------------------------------
      #  self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        # -------------------------- modifided from MAE ------------------------------------

        trunc_normal_(self.cls_token, std=.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)
        if self.pre_norm:
            self.norm = LayerNormSuper(super_embed_dim=embed_dim)

        # classifier head
       # self.head = LinearSuper(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        # todo -------- all added from MAE for deocoding part ------
        self.decoder_embed = LinearSuper(embed_dim, decoder_embed_dim, bias=True) #todo it has to be Supe type to adapt to current sampled_embed_dim

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))  # 1, 1, 512

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                video_vit.Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=None,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.t_pred_patch_size * patch_size ** 2 * in_chans,
            bias=True,
        )
        # todo -------- all added from MAE for deocoding part ------
        # todo: va lasciato??
        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()

    def initialize_weights(self): #todo ----- from MAE
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        w = self.patch_embed_super.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
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
        print('No classifier during pre training') #todo pretrain
     #   return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        return None #to pre train
       # self.num_classes = num_classes
     #   self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def set_sample_config(self, config: dict):
        self.sample_embed_dim = config['embed_dim']
        self.sample_mlp_ratio = config['mlp_ratio']
        self.sample_layer_num = config['layer_num']
        self.sample_num_heads = config['num_heads']
        self.sample_dropout = calc_dropout(self.super_dropout, self.sample_embed_dim[0], self.super_embed_dim)
        self.patch_embed_super.set_sample_config(self.sample_embed_dim[0])
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
            # the decoder_embed dim does not change, I am not looking for decoder architecture
        self.decoder_embed.set_sample_config(sample_in_dim=self.sample_embed_dim[-1], sample_out_dim=self.decoder_embed_dim )
       # self.head.set_sample_config(self.sample_embed_dim[-1], self.num_classes) todo pre train

    def get_sampled_params_numel(self, config):
        self.set_sample_config(config)
        numels = []
        for name, module in self.named_modules():
            if hasattr(module, 'calc_sampled_param_num'):
                if name.split('.')[0] == 'blocks' and int(name.split('.')[1]) >= config['layer_num']:
                    continue
                numels.append(module.calc_sampled_param_num())

        return sum(numels) + self.sample_embed_dim[0]* (2 +self.patch_embed_super.num_patches)
    def get_complexity(self, sequence_length):
        total_flops = 0
        total_flops += self.patch_embed_super.get_complexity(sequence_length)
        total_flops += np.prod(self.pos_embed[..., :self.sample_embed_dim[0]].size()) / 2.0
        for blk in self.blocks:
            total_flops +=  blk.get_complexity(sequence_length+1)
    #    total_flops += self.head.get_complexity(sequence_length+1) todo no classifier during pre training
        return total_flops
    # ------------------------from MAE ---------------------------------
    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, T, H, W = imgs.shape
        p = self.patch_embed_super.patch_size[0]
        u = self.t_pred_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, 3, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 3))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, 3))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 3, T, H, W))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep


    def forward_encoder(self, x, mask_ratio):
        B = x.shape[0]
        x = self.patch_embed_super(x)

        # --- added from mae
        N, T, L, C = x.shape  # C are channels, L is H*W
        x = x.reshape(N, T * L, C)  # now x is all flattened across 3 dimensions
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        x = x.view(N, -1, C)  # 16, 156, 216

        #cls_tokens = self.cls_token[..., :self.sample_embed_dim[0]].expand(B, -1, -1)
        if self.cls_embed:
            cls_token = self.cls_token[..., :self.sample_embed_dim[0]]
            cls_tokens = cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        #if self.abs_pos:
         #   x = x + self.pos_embed[..., :self.sample_embed_dim[0]]
        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        x = x.view([N, -1, C]) + pos_embed[..., :self.sample_embed_dim[0]] #todo devo controllare che i positional
        #embedding facciano giusto metchando sia MAE code che AF code!

        x = F.dropout(x, p=self.sample_dropout, training=self.training)

        # start_time = time.time()
        for blk in self.blocks:
            x = blk(x)
        # print(time.time()-start_time)
        if self.pre_norm:
            x = self.norm(x) #(8, 157, 216) vs in MAE --> (8, 157, 1024)

        if self.gp: #x[:, 1:].shape = torch.Size([8, 156, 216])
            return torch.mean(x[:, 1:] , dim=1), mask, ids_restore

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore # --> MAE outputs x = (8,156, 1024), mask = 8, 1568, ids = 8, 1568

    def forward_decoder(self, x, ids_restore):
        N = x.shape[0]
        T = self.patch_embed_super.t_grid_size
        H = W = self.patch_embed_super.grid_size

        # embed tokens
        #todo you are using as encoding a subnetwork with a sampled_embed_dim. Therefor you must correctly map
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])
        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.repeat(
                1, self.input_size[0], 1
            ) + torch.repeat_interleave(
                self.decoder_pos_embed_temporal,
                self.input_size[1] * self.input_size[2],
                dim=1,
            )
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        x = x + decoder_pos_embed

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, T, H, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        _imgs = torch.index_select(
            imgs,
            2,
            torch.linspace(
                0,
                imgs.shape[2] - 1,
                self.pred_t_dim,
            )
            .long()
            .to(imgs.device),
        )  # 16, 3, 8, 224, 224. 8 is from pred_t_dim defined at 8
        target = self.patchify(_imgs)  # 16, 1568, 768
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss



    def forward(self, x,  mask_ratio=0.75):
       # x = self.forward_features(x)
      #  x = self.head(x)
     #   return x
       latent, mask, ids_restore = self.forward_encoder(x, mask_ratio)
       pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
       loss = self.forward_loss(x, pred, mask)
       return loss, pred, mask


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





