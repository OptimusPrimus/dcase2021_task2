import os
import socket
import numpy as np
import sklearn
import torch
from tqdm import tqdm
from glob import glob


from experiments.data_modules import MCMDataModule
from experiments.feature_extractor import MelSpectrogram


def get_logging_uri(**kwargs):
    return f"file:{kwargs['log_root']}/mlruns"


def get_logging_root():
    return os.path.join(os.getcwd(), 'logs')


def get_data_module(clazz=True, **kwargs):
    if kwargs['data_module'] in ['mcm']:
        if clazz:
            return MCMDataModule
        else:
            return MCMDataModule(**kwargs)
    else:
        raise ValueError(f"Data Module '{kwargs['data_module']}' unknown.")


def get_feature_extractor(**kwargs):
    if kwargs['feature_extractor'] == 'mel_spectrogram':
        return MelSpectrogram(**kwargs)
    else:
        raise ValueError(f"Feature Extractor '{kwargs['feature_extractor']}' unknown.")


def get_best_model_path(dir, mode=None):
    if mode == 'max':
        idx = -1
    elif mode == 'min':
        idx = 0
    else:
        return os.path.join(dir, 'last.ckpt')
    models = list(glob(os.path.join(dir, '**.ckpt')))
    models = sorted(models, key=lambda x: float(os.path.splitext(os.path.split(x)[1])[0].split('=')[-1]))
    print(f'Loading {models[idx]}')
    return models[idx]


def get_experiment(**args):
    from experiments.experiment_classification import ClassificationExperiment
    from experiments.experiment_density import DensityEstimate

    if args['experiment'] == 'multi_section':
        kwargs = {
                'task': 'multi_section',
                'batch_size': 16,
                'snippet_length': 10,
                'snippet_hop_size': 10,

        }
        return ClassificationExperiment, kwargs
    elif args['experiment'] == 'density':
        kwargs = {
            'batch_size': 512,
            'snippet_length': 0.192,
            'snippet_hop_size': 0.032,
            'learning_rate': 0.001,
        }
        return DensityEstimate, kwargs

    else:
        raise ValueError(f"Experiment '{args['experiment']}' unknown.")


def get_mean_std(kwargs, cuda=True):

    encoder = get_feature_extractor(**kwargs)

    if cuda:
        encoder = encoder.cuda()

    scaler = sklearn.preprocessing.StandardScaler()

    dataset = get_data_module(**kwargs)(**kwargs).normal_set(train=True, randomize=False, normalize=True)

    if kwargs['debug']:
        dataset = torch.utils.data.Subset(dataset, list(range(10)))

    pbar = tqdm(dataset)

    for sample in pbar:
        tracks = sample['audio']
        tracks = torch.from_numpy(tracks)

        if cuda:
            tracks = tracks.cuda()

        tracks = encoder(tracks.float()).permute(0, 2, 1)
        tracks = tracks.reshape(-1, tracks.shape[-1])

        pbar.set_description("Compute dataset statistics")
        scaler.partial_fit(np.squeeze(tracks.cpu().numpy()))

    std = np.maximum(scaler.scale_, 1e-4 * np.max(scaler.scale_))

    return scaler.mean_, std


def stack_numpy(k, outputs):
    if k == 'path':
        import itertools
        return list(itertools.chain.from_iterable([o[k] for o in outputs]))
    else:
        return np.concatenate([o[k].detach().cpu().numpy() for o in outputs])


def column_names(k, data):
    if type(data) is np.ndarray and data.ndim == 2:
        n_columns = data.shape[1]
        return [f'{k}_{i}' for i in range(n_columns)]
    else:
        return [k]


def stack_torch(k, batches):
    if type(batches[0][k]) == list:
        import itertools
        return list(itertools.chain.from_iterable([b[k] for b in batches]))
    else:
        return torch.cat([b[k] for b in batches])

