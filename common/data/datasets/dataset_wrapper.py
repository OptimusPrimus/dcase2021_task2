import torch
import librosa
import numpy as np
from common.data.load_audio import load_audio
from tqdm import tqdm
import math
from common.data.datasets.dcase2021_task2 import MACHINE_TYPES

DEFAULT_INCLUDE = {
                'machine_type': MACHINE_TYPES,
                'section': [0, 1, 2, 3, 4, 5],
                'source': [True, False],
                'train': [True, False],
                'anomaly': [True, False]
            }

DEFAULT_EXCLUDE = {

}


class RAMDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset
    ) -> None:
        self.samples = []
        for s in tqdm(dataset, desc='Loading Dataset into RAM...'):
            self.samples.append(s)

    def __getitem__(self, item: int) -> dict:
        return self.samples[item]

    def __len__(self) -> int:
        return len(self.samples)


class LoadAudioDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            sampling_rate: int = 16000
    ) -> None:
        self.sampling_rate = sampling_rate
        self.dataset = dataset

    def __getitem__(self, item: int) -> dict:
        sample = self.dataset[item].copy()
        sample['samplerate'] = self.sampling_rate
        sample['audio'] = load_audio(sample['path'], sampling_rate=self.sampling_rate)
        return sample

    def __len__(self) -> int:
        return len(self.dataset)


class RMSNormalize(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset
    ) -> None:
        self.dataset = dataset

    def __getitem__(self, item: int) -> dict:
        sample = self.dataset[item].copy()
        with np.errstate(divide='raise'):
            try:
                a = np.sqrt(sample['audio'].shape[-1] / np.sum(sample['audio'] ** 2))
                sample['audio'] = sample['audio'] * a
                if sample['audio'].sum() == 0:
                    print(sample['path'])
            except FloatingPointError:
                print(sample['path'])
        return sample

    def __len__(self) -> int:
        return len(self.dataset)

class OnlyPathDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset
    ) -> None:
        self.dataset = dataset

    def __getitem__(self, item: int) -> dict:

        return {
            'path': self.dataset[item]['path'],
            'samplerate': self.dataset[item]['samplerate'],
            'duration': self.dataset[item]['duration']
        }

    def __len__(self) -> int:
        return len(self.dataset)


class FilteredDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            include: dict = DEFAULT_INCLUDE,
            exclude: dict = DEFAULT_EXCLUDE
    ) -> None:
        self.samples = []

        for s in dataset:
            add = True

            for k in include:
                if s[k] not in include[k]:
                    add = False

            for k in exclude:
                if s[k] in exclude.get(k, []):
                    add = False

            if add:
                self.samples.append(s)

    def __getitem__(self, item: int) -> dict:
        return self.samples[item]

    def __len__(self) -> int:
        return len(self.samples)


class MonoDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            axis=0
    ) -> None:
        self.dataset = dataset
        self.axis = axis

    def __getitem__(self, index: int) -> dict:
        sample = self.dataset[index].copy()
        sample['audio'] = sample['audio'].mean(axis=self.axis, keepdims=True)
        return sample

    def __len__(self) -> int:
        return len(self.dataset)


class SnippetDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset: torch.utils.data.Dataset,
            duration: float = 2.08,
            hop_size: float = 0.256,
            pad=True
    ) -> None:

        self.dataset = dataset
        self.duration = duration
        self.hop_size = hop_size
        self.pad = pad

        lengths = []
        indices = []
        modulo = []

        if hop_size == None:
            hop_size = duration

        assert hop_size <= duration

        for i, d in enumerate(dataset):
            if pad:
                length = int(math.ceil((d["duration"] - duration + hop_size) / hop_size))
            else:
                length = int((d["duration"] - duration + hop_size) // hop_size)
            lengths.append(length)
            indices.extend([i]*lengths[-1])
            modulo.extend(list(range(lengths[-1])))

        self.lengths = lengths
        self.indices = indices
        self.modulo = modulo

    def __getitem__(self, index: int) -> dict:

        sample = self.dataset[self.indices[index]].copy()
        snippet_length = int(sample['samplerate'] * self.duration)

        if self.hop_size is None:
            max_offset = int(sample['audio'].shape[-1] - sample['samplerate'] * self.duration)
            offset = torch.randint(max_offset + 1, (1,)).item()
            sample['audio'] = sample['audio'][..., offset:offset + snippet_length]
        else:
            offset = int(self.modulo[index] * self.hop_size * sample['samplerate'])
            sample['part'] = self.modulo[index]
            if offset + snippet_length > sample['audio'].shape[-1]:
                to_pad = sample['audio'][..., offset:]
                sample['audio'] = np.pad(to_pad, ((0,0), (0, snippet_length - to_pad.shape[-1])))
            else:
                sample['audio'] = sample['audio'][..., offset:offset + snippet_length]

        return sample

    def __len__(self) -> int:
        return int(np.array(self.lengths).sum())


class AugmentationDataset(torch.utils.data.Dataset):

    def __init__(
            self,
            dataset,
            pitch_shift=False,
            pariwise_mixing=False,
            gain_augment=False
    ) -> None:

        self.dataset = dataset
        self.pariwise_mixing = pariwise_mixing

        transforms = []
        if gain_augment:
            transforms.append(_augment_gain)
        if pitch_shift:
            transforms.append(_augment_pitch_shift)

        self.aug_single_track = Compose(
            transforms
        )

    def __get_random_sample___(self):
        return self.dataset[torch.randint(len(self), (1,)).item()]

    def __getitem__(self, index) -> dict:

        sample = self.dataset[index].copy()
        sample = self.aug_single_track(sample)

        if self.pariwise_mixing:
            other_sample = self.__get_random_sample___()
            other_sample = self.aug_single_track(other_sample)
            sample = _aug_mix(sample, other_sample)

        return sample

    def __len__(self) -> int:
        return len(self.dataset)


class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms) -> None:
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio


def _augment_gain(sample: dict, low: float = 0.25, high: float = 1.25) -> dict:
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1).item() * (high - low)
    sample['audio'] = sample['audio'] * g
    return sample


def _augment_pitch_shift(sample: dict, low: int = -2, high: int = 2, sr=16000) -> dict:
    """Applies a random gain between `low` and `high`"""
    n_steps = torch.randint(low, high+1, (1,)).item()
    sample['audio'] = librosa.effects.pitch_shift(
        sample['audio'][0, :].astype(np.float32), sr, n_steps=n_steps
    )[None, :].astype(np.float16)

    sample['pitch_shift'] = n_steps

    return sample


def _aug_mix(sample: dict, other_sample: dict) -> dict:
    # r1, r2 ... limits of aug_factor
    sample = sample.copy()
    mix_factor = torch.rand(1).item()
    sample['audio'] = sample['audio'] * mix_factor + other_sample['audio'] * (1-mix_factor)

    sample['mix_factor'] = mix_factor
    sample['other_section'] = other_sample['section']
    if other_sample.get('pitch_shift'):
        sample['other_pitch_shift'] = other_sample['pitch_shift']

    return sample


def _augment_channelswap(audio: np.array) -> np.array:
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.rand(1).item() < 0.5:
        return np.flip(audio, 0)
    else:
        return audio

def aug_from_str(list_of_function_names: list):
    if list_of_function_names:
        return Compose([globals()["_augment_" + aug] for aug in list_of_function_names])
    else:
        return lambda audio: audio


