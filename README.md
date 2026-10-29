# NAS_4IV: One-Shot Neural Architecture Search for Vision Transformers (Images and Video)

This repository contains research code for one-shot neural architecture search (NAS) over vision transformers, extending the weight-entanglement supernet approach of [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer) from images to video. It is associated with the paper "NAS just once".

<!-- PAPER SECTION -->
<!-- TODO: paper title, venue, authors, links, figures (put images in assets/) and BibTeX citation. -->

## Contents

- **`Tubes_AF/`**: AutoFormer-style supernet for video. Adds a sparse-tubes tokenizer (`SparseTubesTokenizerSuper`), 3D sin-cos positional encodings (`tubevit/`), a self-attention-pooling head, and single/multi-dataset supernet training (`supernet_train.py`, `supernet_multi_train.py`) plus evolutionary search (`evolution.py`). Configs in `experiments/`.
- **`Masked_AutoFormer/`**: masked-autoencoder-style supernet (`supernet_autoencoder.py`, `video_vit.py`) with pre-training (`supernet_pretrain.py`), fine-tuning (`supernet_finetune.py`) and evolutionary search; Kinetics video data pipeline in `util/`.
- **`third_party/cream/`**: unmodified copy of [microsoft/Cream](https://github.com/microsoft/Cream) (AutoFormer, AutoFormerV2, CDARTS, Cream, EfficientViT, MiniViT, TinyCLIP, TinyViT), kept for reference and baselines. Each sub-project has its own README and is self-contained.
- **`assets/`**: figures for the README.

## Repository structure

```
.
├── Tubes_AF/            # video supernet (sparse tubes) + search
│   ├── lib/  model/  tubevit/  experiments/
│   └── supernet_train.py  supernet_multi_train.py  supernet_engine.py  evolution.py
├── Masked_AutoFormer/   # masked-autoencoder supernet + search
│   ├── lib/  model/  util/  experiments/
│   └── supernet_pretrain.py  supernet_finetune.py  evolution.py
├── third_party/cream/   # upstream Microsoft Cream projects (MIT)
├── assets/
└── LICENSE
```

## Installation

```bash
conda create -n nas4iv python=3.8
conda activate nas4iv
pip install -r Tubes_AF/requirements.txt        # or Masked_AutoFormer/requirements.txt
```

`Tubes_AF` imports `Masked_AutoFormer.util.kinetics`, so both folders must stay side by side; the scripts add the repository root to `sys.path` automatically.

## Usage

Run scripts from inside the project folder (they use project-local imports such as `lib`, `model`). Exact argument values depend on your data and hardware; see `--help` for all options.

```bash
cd Tubes_AF
python supernet_train.py --cfg experiments/supernet/supernet-T.yaml --data-path /path/to/video_data --data-set k600 ...
python supernet_multi_train.py --cfg experiments/supernet/supernet-T.yaml --data-path /path/a /path/b ...
python evolution.py --cfg experiments/subnet/AutoFormer-T.yaml ...

cd ../Masked_AutoFormer
python supernet_pretrain.py --cfg experiments/supernet/supernet-T.yaml ...
python supernet_finetune.py --cfg experiments/supernet/supernet-T.yaml ...
```

## Data

- `Tubes_AF`: video datasets (default path in the code points to a Kinetics-600 layout; `--data-path` accepts several paths for multi-dataset training). Image datasets (ImageNet, CIFAR, etc.) are supported through `lib/datasets.py`.
- `Masked_AutoFormer`: Kinetics-style video folders and ImageNet (`--data-set IMNET`).

## Acknowledgements

This work builds on [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer) and the rest of [microsoft/Cream](https://github.com/microsoft/Cream) (Microsoft Corporation, MIT License), and the masked-autoencoder video code derives from Meta Platforms' MAE-style code (see file headers). Original third-party READMEs, licenses and the Microsoft code of conduct/security policy are preserved under `third_party/cream/`.

## License

MIT; see [LICENSE](LICENSE) (copyright Microsoft Corporation for the upstream code).
