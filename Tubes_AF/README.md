# Tubes_AF

Main VIM-NAS code: a weight-entangled, AutoFormer-style supernet that processes images and videos through 2D patches and 3D sparse space-time tubes, with joint multi-dataset training and evolutionary search. See the [top-level README](../README.md) for the paper and overview.

## Layout

- `model/`: supernet transformer (`supernet_transformer.py`) and super-layers (`module/`, including `SparseTubesTokenizerSuper` in `embedding_super.py`).
- `tubevit/`: 2D/3D sin-cos positional encodings.
- `lib/`: configs, datasets (`datasets.py`, `multi_datasets.py`), samplers (including the dataset sampling for joint training), utilities.
- `experiments/`: supernet and subnet YAML configs (search space: embedding dim, heads, MLP ratio, depth).
- `supernet_train.py`, `supernet_multi_train.py`, `supernet_engine.py`, `evolution.py`: training and search entry points.

## Setup

```bash
pip install -r requirements.txt
```

`Masked_AutoFormer/` must stay next to this folder (the Kinetics loader is imported from it).

## Usage

Run from inside `Tubes_AF/`; see `--help` for all options.

```bash
python supernet_train.py --cfg experiments/supernet/supernet-T.yaml --data-path /path/to/video_data --data-set k600 ...
python supernet_multi_train.py --cfg experiments/supernet/supernet-T.yaml --data-path /path/a /path/b ...
python evolution.py --cfg experiments/subnet/AutoFormer-T.yaml ...
```

## Credits

Built on [AutoFormer](https://github.com/microsoft/Cream/tree/main/AutoFormer) from [microsoft/Cream](https://github.com/microsoft/Cream) (MIT License).
