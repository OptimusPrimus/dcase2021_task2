import numpy as np
import librosa


def load_info(path: str) -> dict:
    """Load audio metadata

    this is a backend_independent wrapper around torchaudio.info

    Args:
        path: Path of filename
    Returns:
        Dict: Metadata with
        `samplerate` in milliseconds, `duration` in seconds

    """

    info = {}
    info['samplerate'] = librosa.get_samplerate(str(path))
    info['duration'] = librosa.get_duration(filename=str(path))
    info['path'] = str(path)
    return info


def load_audio(path: str, sampling_rate: int = None, mono: bool = True) -> np.ndarray:
    """Load audio file

    Args:
        path: Path of audio file
    Returns:
        Array: numpy tensor waveform of shape `(num_channels, num_samples)`
    """

    sig = librosa.load(path, sr=sampling_rate, mono=mono, dtype=np.float32)[0]
    if len(sig.shape) == 1:
        sig = sig.reshape(1, -1)
    return sig
