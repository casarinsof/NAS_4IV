import torch
from Masked_AutoFormer.util.kinetics import Kinetics
import os
from torchvision import datasets
import numpy as np
# NON POSSO AVERE IL TRAIN_dS NEL GET_ITEM PERCHE SENNO MI DA TRAIN_DS DIVERSI OGNI VOLTA
class MultiDatasets(torch.utils.data.Dataset):
    def __init__(self,
                 args,
                 # ----- imagenet -----
                 image_root,
                 image_transforms,
                 is_train,
                 # ---- kinetics ----
                 mode,
                 video_root,
                 # decoding setting
                 sampling_rate=4,
                 num_frames=16,
                 target_fps=30,
                 # train aug settings
                 train_jitter_scales=(256, 320),
                 train_crop_size=224,
                 train_random_horizontal_flip=True,
                 # test setting, multi crops
                 test_num_ensemble_views=10,
                 test_num_spatial_crops=3,
                 test_crop_size=256,
                 # norm setting
                 mean=(0.45, 0.45, 0.45),
                 std=(0.225, 0.225, 0.225),
                 # other parameters
                 enable_multi_thread_decode=False,
                 use_offset_sampling=True,
                 inverse_uniform_sampling=False,
                 num_retries=10,
                 # pretrain augmentation
                 repeat_aug=1,
                 aa_type="rand-m7-n4-mstd0.5-inc1",
                 pretrain_rand_flip=True,
                 pretrain_rand_erase_prob=0.25,
                 pretrain_rand_erase_mode="pixel",
                 pretrain_rand_erase_count=1,
                 pretrain_rand_erase_split=False,
                 rand_aug=False,
                 jitter_scales_relative=[0.5, 1.0],
                 jitter_aspect_relative=[0.75, 1.3333],
                 ):
        self.kinetics = Kinetics(mode=mode, path_to_data_dir=video_root, sampling_rate=sampling_rate,
                                 num_frames=num_frames, train_jitter_scales=train_jitter_scales, repeat_aug=repeat_aug,
                                 pretrain_rand_erase_prob=pretrain_rand_erase_prob,
                                 pretrain_rand_erase_mode=pretrain_rand_erase_mode,
                                 pretrain_rand_erase_count=pretrain_rand_erase_count,
                                 pretrain_rand_erase_split=pretrain_rand_erase_split,
                                 aa_type=aa_type, rand_aug=rand_aug,
                                 jitter_aspect_relative=jitter_aspect_relative,
                                 jitter_scales_relative=jitter_scales_relative,
                                )
        self.img_root = os.path.join(image_root, 'train' if is_train else 'val')
        self.imagenet = datasets.ImageFolder(self.img_root, transform=image_transforms)
    #    self.image_transforms = image_transforms

        steps_per_epoch0 = len(self.kinetics) // args.batch_size
        steps_per_epoch1 = len(self.imagenet) // args.batch_size

        self.step0 = steps_per_epoch0
        self.step1 = steps_per_epoch1

