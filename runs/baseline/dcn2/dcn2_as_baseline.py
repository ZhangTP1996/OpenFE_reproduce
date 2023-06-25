# %%
import math
import typing as ty
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import zero
import os
import lib
import argparse


class CrossLayer(nn.Module):
    """ Cross Layer module for DCNv2,
        follow implementation of https://github.com/Yura52/tabular-dl-revisiting-models/bin/dcn2.py
    """
    def __init__(self, d, dropout):
        super().__init__()
        self.linear = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0, x):
        return self.dropout(x0 * self.linear(x)) + x

class DCNv2(nn.Module):
    """ Deep & Cross Network v2,
        follow implementation of https://github.com/Yura52/tabular-dl-revisiting-models/bin/dcn2.py
    """
    def __init__(
        self,
        *,
        d_in: int,
        d: int,
        n_hidden_layers: int,
        n_cross_layers: int,
        hidden_dropout: float,
        cross_dropout: float,
        d_out: int,
        stacked: bool,
        categories: ty.Optional[ty.List[int]],
        d_embedding: int,
    ) -> None:
        super().__init__()

        if categories is not None:
            d_in += len(categories) * d_embedding
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            self.category_embeddings = nn.Embedding(sum(categories), d_embedding)
            nn.init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))
            print(f'{self.category_embeddings.weight.shape=}')

        self.first_linear = nn.Linear(d_in, d)
        self.last_linear = nn.Linear(d if stacked else 2 * d, d_out)

        deep_layers = sum(
            [
                [nn.Linear(d, d), nn.ReLU(True), nn.Dropout(hidden_dropout)]
                for _ in range(n_hidden_layers)
            ],
            [],
        )
        cross_layers = [CrossLayer(d, cross_dropout) for _ in range(n_cross_layers)]

        self.deep_layers = nn.Sequential(*deep_layers)
        self.cross_layers = nn.ModuleList(cross_layers)
        self.stacked = stacked

    def forward(self, x_num, x_cat):
        x = []
        if x_num is not None:
            x.append(x_num)
        if x_cat is not None:
            x.append(
                self.category_embeddings(x_cat + self.category_offsets[None]).view(
                    x_cat.size(0), -1
                )
            )
        x = torch.cat(x, dim=-1)

        x = self.first_linear(x)
        print(x.shape)

        x_cross = x
        for cross_layer in self.cross_layers:
            x_cross = cross_layer(x, x_cross)

        if self.stacked:
            return self.last_linear(self.deep_layers(x_cross)).squeeze(1)
        else:
            return self.last_linear(
                torch.cat([x_cross, self.deep_layers(x)], dim=1)
            ).squeeze(1)

def run_dcn2_as_baseline(args_file, args, output):
    """ Run DCNv2 as a feature generation baseline,
        only output the extracted features before the last linear layer,
        follow parallel structure of dcn2.
    """
    def forward_pre_hook(module, input):
        """ Hook to save the input of the last linear layer"""
        print(input[0].detach().numpy().shape)
        new_features.append(input[0].detach().numpy())

    zero.set_randomness(args['seed'])
    dataset_dir = lib.get_path(args['data']['path'])
    stats: ty.Dict[str, ty.Any] = {
        'dataset': dataset_dir.name,
        'algorithm': Path(__file__).stem,
        **lib.load_json(output / 'stats.json'),
    }
    timer = zero.Timer()
    timer.run()

    D = lib.Dataset.from_dir(dataset_dir)
    X = D.build_X(
        normalization=args['data'].get('normalization'),
        num_nan_policy='mean',
        cat_nan_policy='new',
        cat_policy=args['data'].get('cat_policy', 'indices'),
        cat_min_frequency=args['data'].get('cat_min_frequency', 0.0),
        seed=args['seed'],
    )
    if not isinstance(X, tuple):
        X = (X, None)

    zero.set_randomness(args['seed'])
    Y, y_info = D.build_y(args['data'].get('y_policy'))
    lib.dump_pickle(y_info, output / 'y_info.pickle')
    X = tuple(None if x is None else lib.to_tensors(x) for x in X)
    Y = lib.to_tensors(Y)
    device = lib.get_device()
    if device.type != 'cpu':
        X = tuple(None if x is None else {k: v.to(device) for k, v in x.items()} for x in X)
        Y_device = {k: v.to(device) for k, v in Y.items()}
    else:
        Y_device = Y
    X_num, X_cat = X
    if not D.is_multiclass:
        Y_device = {k: v.float() for k, v in Y_device.items()}

    train_size = D.size(lib.TRAIN)
    batch_size = args['training']['batch_size']
    epoch_size = stats['epoch_size'] = math.ceil(train_size / batch_size)

    loss_fn = (
        F.binary_cross_entropy_with_logits
        if D.is_binclass
        else F.cross_entropy
        if D.is_multiclass
        else F.mse_loss
    )
    args['model'].setdefault('d_embedding', None)
    model = DCNv2(
        d_in=0 if X_num is None else X_num['train'].shape[1],
        d_out=D.info['n_classes'] if D.is_multiclass else 1,
        categories=lib.get_categories(X_cat),
        **args['model'],
    ).to(device)

    stats['n_parameters'] = lib.get_n_parameters(model)
    optimizer = lib.make_optimizer(
        args['training']['optimizer'],
        model.parameters(),
        args['training']['lr'],
        args['training']['weight_decay'],
    )

    stream = zero.Stream(lib.IndexLoader(train_size, batch_size, True, device))
    progress = zero.ProgressTracker(args['training']['patience'])
    training_log = {lib.TRAIN: [], lib.VAL: [], lib.TEST: []}
    timer = zero.Timer()
    checkpoint_path = lib.env.PROJECT_DIR / f'{args_file.data}/{args_file.model_type}/checkpoint.pt'

    # load trained model from NN experiment
    model.load_state_dict(torch.load(checkpoint_path)['model'])
    model.eval()

    new_features = []

    # register hook to save new train, val, test features, respectively
    model.last_linear.register_forward_pre_hook(forward_pre_hook)
    for part in ['train', 'val', 'test']:
        output = model(
                        None if X_num is None else X_num[part],
                        None if X_cat is None else X_cat[part])   

    model.last_linear.register_forward_pre_hook(None)
    print('new feature shape:', new_features[0].shape, new_features[1].shape, new_features[2].shape)

    return new_features[0], new_features[1], new_features[2]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='default')
    parser.add_argument('--orig_path', type=str, required=True)  # path to original NN experiment dir
    args_file = parser.parse_args()
    # prepare config
    os.system(f'mkdir -p {args_file.data}/{args_file.model_type}/')
    os.system(f'cp {args_file.orig_path}/output/{args_file.data}/dcn2/{args_file.model_type}/0.toml {args_file.data}/{args_file.model_type}/0.toml')
    
    p = Path(f'{args_file.data}/{args_file.model_type}/0.toml')
    p.write_text(p.read_text().replace(f"path = 'data/{args_file.data}'",
                                        f"path = '{args_file.orig_path}/data/{args_file.data}'"))
    # load or create checkpoint
    if os.path.exists(f'{args_file.orig_path}/output/{args_file.data}/dcn2/{args_file.model_type}/0/checkpoint.pt'):
        os.system(f'cp {args_file.orig_path}/output/{args_file.data}/dcn2/{args_file.model_type}/0/checkpoint.pt {args_file.data}/{args_file.model_type}/checkpoint.pt')
    else:
        res = os.system(
                f'CUDA_VISIBLE_DEVICES=0,1 python {args_file.orig_path}/bin/dcn2.py --force {args_file.data}/{args_file.model_type}/0.toml')
        os.system(f'mv {args_file.data}/{args_file.model_type}/0/checkpoint.pt {args_file.data}/{args_file.model_type}/checkpoint.pt')
    
    # get new features
    args, output = lib.load_config_from_path(f'{args_file.data}/{args_file.model_type}/0.toml')
    new_train, new_val, new_test = run_dcn2_as_baseline(args_file, args, output)

    new_path = f'{args_file.orig_path}/data/{args_file.data}_dcn2'
    os.system(f'cp -r {args_file.orig_path}/data/{args_file.data} {new_path}')

    # save and concat new features
    for part in ['train', 'val', 'test']:
        original_feature = np.load(new_path + f'/N_{part}.npy')
        new_feature = locals()[f'new_{part}']
        assert(original_feature.shape[0] == new_feature.shape[0])
        new_data = np.concatenate([original_feature, new_feature], axis=1)
        print(f'new {part} shape:', new_data.shape)
        np.save(new_path + f'/N_{part}.npy', new_data)

    print('done')
