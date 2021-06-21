from common.data.datasets import LoadAudioDataset, RAMDataset, Compose, FilteredDataset, MachineDataSet, \
    AugmentationDataset, RMSNormalize, SnippetDataset
from common.data.datasets.dcase2021_task2 import MACHINE_TYPES, NUM_SECTIONS

import torch

import pytorch_lightning as pl
from argparse import ArgumentParser
from torch.utils.data import Subset, DataLoader, ConcatDataset


class MCMDataModule(pl.LightningDataModule):

    def __init__(
            self,
            machine_type: str,
            section: int,
            data_root: str,
            batch_size: int = 16,
            n_workers=4,
            normalize_waveform = False,
            proxy_outliers='none',
            debug=False,
            verbose=True,
            **kwargs
    ) -> None:

        super().__init__()

        if section == None:
            self.sections = [0, 1, 2, 3, 4, 5]
        elif section == -1:
            self.sections = [0, 1, 2]
        elif 0 <= section < 6:
            self.sections = [section]
        else:
            raise AttributeError

        self.machine_types = [machine_type] if (machine_type is not None) else MACHINE_TYPES

        self.data_root = data_root
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.kwargs = kwargs
        self.kwargs['proxy_outliers'] = proxy_outliers
        self.verbose = verbose

        transforms = []
        if debug:
            transforms.append(lambda x: Subset(x, list(range(batch_size*3))))
            self.n_workers = 0
        transforms.append(lambda x: LoadAudioDataset(x, sampling_rate=kwargs.get('sampling_rate', 16000)))
        if normalize_waveform:
            transforms.append(RMSNormalize)
        transforms.append(RAMDataset)
        transforms.append(lambda x: SnippetDataset(x, duration=kwargs['snippet_length'], hop_size=None))
        self.train_transform = Compose(transforms)
        transforms = transforms.copy()
        transforms.pop(-1)
        transforms.append(lambda x: SnippetDataset(x, duration=kwargs['snippet_length'], hop_size=kwargs['snippet_hop_size']))
        self.val_transform = Compose(transforms)
        transforms = transforms.copy()
        transforms.pop(-1)
        self.norm_transform = Compose(transforms)

    def normal_set(self, train=True, randomize=True, source=None, normalize=False) -> torch.utils.data.Dataset:

        if source == None:
            source = [True, False]
        else:
            source = [source]
        normal_set = FilteredDataset(
            MachineDataSet(data_root=self.data_root),
            include={
                'machine_type': self.machine_types,
                'section': self.sections,
                'source': source,
                'train': [train],
                'anomaly': [True, False, -1]
            }
        )
        if normalize:
            return self.norm_transform(normal_set)
        elif randomize:
            return self.train_transform(normal_set)
        else:
            return self.val_transform(normal_set)

    def outlier_set(self) -> torch.utils.data.Dataset:
        if self.kwargs['proxy_outliers'] == 'other_sections':

            if len(self.sections) == 1 or len(self.sections) == 3:
                other_sections = [i for i in range(NUM_SECTIONS) if i not in self.sections]
            else:
                other_sections = list(range(NUM_SECTIONS))

            assert len(self.machine_types) == 1
            outlier_set = FilteredDataset(
                MachineDataSet(data_root=self.data_root),
                include={
                    'machine_type': self.machine_types,
                    'section': other_sections,
                    'source': [True, False],
                    'train': [True],
                    'anomaly': [False]
            })
        elif self.kwargs['proxy_outliers'] == 'other_sections_and_machines':
            assert len(self.machine_types) == 1
            outlier_set = FilteredDataset(
                MachineDataSet(data_root=self.data_root),
                include={
                    'machine_type': MACHINE_TYPES,
                    'section': list(range(NUM_SECTIONS)),
                    'source': [True, False],
                    'train': [True],
                    'anomaly': [False]
            })
        elif self.kwargs['proxy_outliers'] == 'other_machines':
            assert len(self.machine_types) == 1
            machines = MACHINE_TYPES.copy()
            for m in self.machine_types:
                machines.remove(m)
            outlier_set = FilteredDataset(
                MachineDataSet(data_root=self.data_root),
                include={
                    'machine_type': machines,
                    'section': list(range(NUM_SECTIONS)),
                    'source': [True, False],
                    'train': [True],
                    'anomaly': [False]
                })
        elif self.kwargs['proxy_outliers'] == 'none':
            return None
        else:
            raise AttributeError(f"Outlier set '{self.kwargs['proxy_outliers']}' not implemented.")

        return self.train_transform(outlier_set)

    def train_dataloader(self):

        source_set = self.normal_set(train=True, randomize=True, source=True)

        if self.kwargs.get('da_task') not in [None, 'none']:
            target_set = self.normal_set(train=True, randomize=True, source=False)
            repetitions = ((len(source_set) // len(target_set)) + 1)
            target_set = ConcatDataset([target_set] * repetitions)

            source_set = DataLoader(
                AugmentationDataset(source_set, pariwise_mixing=self.kwargs['pairwise_mixing']),
                batch_size=self.batch_size,
                num_workers=self.n_workers,
                shuffle=True
            )

            target_set = DataLoader(
                AugmentationDataset(target_set, pariwise_mixing=self.kwargs['pairwise_mixing']),
                batch_size=self.batch_size,
                num_workers=self.n_workers,
                shuffle=True
            )
            train_sets = [source_set, target_set]
        else:
            train_sets = [
                DataLoader(
                    AugmentationDataset(source_set, pariwise_mixing=self.kwargs['pairwise_mixing']),
                    batch_size=self.batch_size,
                    num_workers=self.n_workers,
                    shuffle=True,
                    drop_last=True
                )
            ]

        outlier_set = self.outlier_set()
        if (outlier_set is None) or len(outlier_set) == 0:
            return train_sets

        train_sets.append(DataLoader(outlier_set, batch_size=self.batch_size, num_workers=self.n_workers, shuffle=True, drop_last=True))

        return train_sets

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        val_set = self.normal_set(train=False, randomize=False)
        return DataLoader(val_set, batch_size=self.batch_size, num_workers=self.n_workers)

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        val_set = ConcatDataset([self.normal_set(train=False, randomize=False), self.normal_set(train=True, randomize=False)])
        return DataLoader(val_set, batch_size=self.batch_size, num_workers=self.n_workers)

    @staticmethod
    def add_model_specific_args(parent_parser, **defaults):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--machine_type', type=str, default='fan')
        parser.add_argument('--section', type=int, default=None)
        parser.add_argument('--proxy_outliers', type=str, default='none', choices=['none', 'other_sections', 'other_machines', 'other_sections_and_machines'])
        parser.add_argument('--batch_size', type=int, default=defaults.get('batch_size', 16))
        parser.add_argument('--snippet_length', type=float, default=defaults.get('snippet_length', 10))
        parser.add_argument('--snippet_hop_size', type=float, default=defaults.get('snippet_hop_size', 10))
        parser.add_argument('--pairwise_mixing', dest='pairwise_mixing', action='store_true')
        parser.set_defaults(pairwise_mixing=False)
        parser.add_argument('--normalize_waveform', dest='normalize_waveform', action='store_true')
        parser.set_defaults(normalize_waveform=False)
        parser.add_argument('--n_workers', type=int, default=4)

        return parser
