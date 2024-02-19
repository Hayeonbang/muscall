import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union, Callable
import torch
import torchaudio
from torch import Tensor
from torch.utils.data import Dataset

import json


class TaggingDataset(Dataset):
    """Create a Dataset for music multi-label classification (auto-tagging).
    Args:
        root (str or Path): Path to the directory where the dataset is found.
        audio_transform (Callable): list of transformations to be applied to the audio.
        subset (str, optional): Which subset of the dataset to use.
            One of ``"training"``, ``"validation"``, ``"testing"`` or ``None``.
            If ``None``, the entire dataset is used. (default: ``None``).
    """

    def __init__(
        self,
        root: Union[str, Path],
        audio_transform: Callable = None,
        subset: Optional[str] = "training",
    ) -> None:

        super().__init__()
    
        self.subset = subset

        assert subset is None or subset in ["training", "validation", "testing"], (
            "When `subset` not None, it must take a value from "
            + "{'training', 'validation', 'testing'}."
        )

        self.audio_transform = audio_transform

        self._path = os.fspath(root)

        if not os.path.isdir(self._path):
            raise RuntimeError("Dataset not found.")

    def get_audio_id(self):
        raise NotImplementedError

    def load_audio(self):
        raise NotImplementedError

    def get_tags(self):
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str]:
        """Load the n-th sample from the dataset.

        Args:
            n (int): The index of the sample to be loaded

        Returns:
            tuple: ``(waveform, label)``
        """
        waveform = self.load_audio(n)
        label = self.get_tags(n)

        return waveform, label

    @classmethod
    def num_classes(cls):
        raise NotImplementedError


class MTTDataset(TaggingDataset):
    """Create a Dataset for MagnaTagATune.
    Args:
        root (str or Path): Path to the directory where the dataset is found or downloaded.
        audio_transform (Callable): list of transformations to be applied to the audio.
        subset (str, optional): Which subset of the dataset to use.
            One of ``"training"``, ``"validation"``, ``"testing"`` or ``None``.
            If ``None``, the entire dataset is used. (default: ``None``).
    """

    _ext_audio = ".mp3"

    def __init__(
        self,
        root: Union[str, Path],
        audio_transform: Callable = None,
        subset: Optional[str] = "training",
    ) -> None:

        super().__init__(root, audio_transform, subset)

        if self.subset == "training":
            self.file_list = np.load(os.path.join(root, "train.npy"))
        elif self.subset == "validation":
            self.file_list = np.load(os.path.join(root, "valid.npy"))
        elif self.subset == "testing":
            self.file_list = np.load(os.path.join(root, "test.npy"))
            self.file_list = [
                i
                for i in self.file_list
                if i.split("\t")[1] != "f/tilopa-turkishauch-05-schlicht_1-88-117.mp3"
            ]

        self.binary_labels = np.load(os.path.join(root, "binary.npy"))

    def load_audio(self, n):
        torchaudio.set_audio_backend("sox_io")
        _, file_name = self.file_list[n].split("\t")
        path_to_audio = os.path.join(self._path, "AUDIO", file_name)
        waveform, sample_rate = torchaudio.load(path_to_audio)
        if self.audio_transform is not None:
            waveform = self.audio_transform(waveform)
        waveform = waveform.squeeze()
        length = 20 * 16000
        start = int((waveform.size(0) - length) / 2.0)
        return waveform[start : start + length]

    def get_tags(self, n):
        audio_id, _ = self.file_list[n].split("\t")
        label = self.binary_labels[int(audio_id)]
        return label

    @classmethod
    def num_classes(cls):
        return 50


class TestDataset(Dataset):
    """Custom dataset for loading audio and captions from JSON file."""
    
    def __init__(self, json_path, npy_dir, subset="testing"):
        """
        Args:
            json_path (str): Path to the JSON file containing dataset information.
            npy_dir (str): Directory where numpy files are stored.
            subset (str, optional): Which subset of the dataset to use. Default is "testing".
        """
        assert subset in ["training", "validation", "testing"], "Subset must be one of 'training', 'validation', or 'testing'."

        self.json_path = json_path
        self.npy_dir = npy_dir
        self.subset = subset

        with open(json_path, 'r') as file:
            self.data = json.load(file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        audio_path = Path(self.npy_dir) / item['audio_path']
        waveform = np.load(audio_path)

        # 오디오를 토치 텐서로 변환
        waveform = torch.tensor(waveform, dtype=torch.float)

        # 오디오 길이를 통일하기 위한 처리
        # 예: 오디오를 20초 길이로 자름 (16000 샘플/초)
        length = 20 * 16000
        if waveform.size(0) < length:
            # 짧은 오디오는 패딩
            padding = length - waveform.size(0)
            waveform = torch.nn.functional.pad(waveform, (0, padding), 'constant', 0)
        else:
            # 긴 오디오는 중간 부분을 잘라냄
            start = int((waveform.size(0) - length) / 2.0)
            waveform = waveform[start : start + length]

        caption = item['caption']
        return waveform, caption
    @staticmethod
    def num_classes():
        return 36  # As there are 36 tags

    @staticmethod
    def get_all_tags():
        # Return a list of all tags in the dataset
        # This method needs to be implemented based on how tags are stored and used in your application
        pass