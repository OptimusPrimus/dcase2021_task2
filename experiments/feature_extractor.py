from argparse import ArgumentParser
import torch
import torchaudio
from torchaudio.transforms import AmplitudeToDB
import sys
import librosa


class MelSpectrogram(torch.nn.Module):

    def __init__(
            self,
            sampling_rate=16000,
            hop_length=512,
            n_fft=1024,
            n_mels=128,
            consistent_with_librosa=False,
            **kwargs
    ):
        super(MelSpectrogram, self).__init__()

        mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sampling_rate,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
            center=False
        )

        if consistent_with_librosa:
            mel.mel_scale.fb = torch.from_numpy(librosa.filters.mel(sampling_rate, n_fft, n_mels=n_mels)).T

        to_db = AmplitudeToDB()
        self.features = torch.nn.Sequential(mel, to_db)

    def forward(self, x):
        return self.features(x)

    @staticmethod
    def add_model_specific_args(parent_parser, **defaults):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--n_mels', type=int, default=defaults.get('n_mels', 128))
        parser.add_argument('--n_fft', type=int, default=defaults.get('n_fft', 1024))
        parser.add_argument('--hop_length', type=int, default=defaults.get('hop_length', 512))
        parser.add_argument('--sampling_rate', type=int, default=defaults.get('sampling_rate', 16000))
        parser.add_argument('--consistent_with_librosa', dest='consistent_with_librosa', action='store_true')
        parser.set_defaults(consistent_with_librosa=False)

        return parser