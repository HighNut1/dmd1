from abc import abstractstaticmethod

# import wsq  # needed for loading nist sd 14 dataset
import torch
import torchvision.transforms.functional as VTF
import cv2
import os
import numpy as np

from flx.data.image_helpers import (
    pad_and_resize_to_deepprint_input_size,
)
from flx.data.dataset import Identifier, IdentifierSet, DataLoader
from flx.data.file_index import FileIndex


class ImageLoader(DataLoader):
    def __init__(self, root_dir: str):
        self._files: FileIndex = FileIndex(
            root_dir, self._extension(), self._file_to_id_fun
        )

    @property
    def ids(self) -> IdentifierSet:
        return self._files.ids

    def get(self, identifier: Identifier) -> torch.Tensor:
        return self._load_image(self._files.get(identifier))

    @abstractstaticmethod
    def _extension() -> str:
        pass

    @abstractstaticmethod
    def _file_to_id_fun(subdir: str, filename: str) -> Identifier:
        pass

    @abstractstaticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        pass


class SFingeLoader(ImageLoader):
    @staticmethod
    def _extension() -> str:
        return ".png"

    @staticmethod
    def _file_to_id_fun(_: str, filename: str) -> Identifier:
        # Pattern: <dir>/<subject_id>_<impression_id>.png
        subject_id, impression_id = filename.split("_")
        # We must start indexing at 0 instead of 1 to be compatible with pytorch
        return Identifier(int(subject_id) - 1, int(impression_id) - 1)

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath, flags=cv2.IMREAD_GRAYSCALE)
        return VTF.to_tensor(img[:-32])


class FVC2004Loader(ImageLoader):
    @staticmethod
    def _extension() -> str:
        return ".tif"

    @staticmethod
    def _file_to_id_fun(filename: str) -> Identifier:
        # Pattern: <dir>/<subject_id>_<sample_id>.png
        subject_id, impression_id = filename.split("_")
        # We must start indexing at 0 instead of 1 to be compatible with pytorch
        return Identifier(int(subject_id) - 1, int(impression_id) - 1)

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return pad_and_resize_to_deepprint_input_size(img, fill=1.0)


class MCYTOpticalLoader(ImageLoader):
    @staticmethod
    def _extension() -> str:
        return ".bmp"

    @staticmethod
    def _file_to_id_fun(_: str, filename: str) -> Identifier:
        # Pattern: <dir>/<person>_<finger>_<impression>.png
        _, person, finger, impression = filename.split("_")
        # 12 impressions per finger
        subject = (10 * int(person)) + int(finger)
        return Identifier(subject, int(impression))

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        width = img.shape[1]
        return pad_and_resize_to_deepprint_input_size(img, roi=(310, width), fill=1.0)


class MCYTCapacitiveLoader(ImageLoader):
    @staticmethod
    def _extension() -> str:
        return ".bmp"

    @staticmethod
    def _file_to_id_fun(_: str, filename: str) -> Identifier:
        # Pattern: <dir>/<person:04d>_<finger>_<impression>.png
        _, person, finger, impression = filename.split("_")
        # 12 impressions per finger
        subject = (10 * int(person)) + int(finger)
        return Identifier(subject, int(impression))

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return pad_and_resize_to_deepprint_input_size(img, fill=1.0)


class NistSD4Dataset(ImageLoader):
    @staticmethod
    def _extension() -> str:
        return ".png"

    @staticmethod
    def _file_to_id_fun(_: str, filename: str) -> Identifier:
        # Pattern: <dir>/[f|s]<subject:04d>_<finger:02d>.png
        sample = 0 if filename[0] == "f" else 1
        subject, _ = filename[1:].split("_")
        # We must start indexing at 0 instead of 1 to be compatible with pytorch
        return Identifier(
            subject=int(subject) - 1,
            impression=sample,
        )

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        return pad_and_resize_to_deepprint_input_size(img, fill=1.0)


class Nist14Loader(ImageLoader):
    @staticmethod
    def _extension() -> str:
        return ".bmp" 

    @staticmethod
    def _file_to_id_fun(subdir: str, filename: str) -> Identifier:
        # 文件名示例: F0000001.bmp 或 S0000001.bmp
        name = filename.split('.')[0]
        prefix = name[0]  # 'F' 或 'S'
        
        # 提取后面的数字作为 subject_id，并减 1 转换为从 0 开始的 PyTorch 兼容索引
        subject_id = int(name[1:]) - 1  
        
        # F库(Gallery)作为 impression 0，S库(Probe)作为 impression 1
        impression_id = 0 if prefix == 'F' else 1
        
        return Identifier(subject=subject_id, impression=impression_id)

    @staticmethod
    def _load_image(filepath: str) -> torch.Tensor:
        # 1. 检查文件是否存在（防止手动删除或文件丢失导致崩溃）
        if not os.path.exists(filepath):
            print(f"\n[警告] 文件不存在，使用纯白占位图跳过: {filepath}")
            dummy_img = np.ones((512, 512), dtype=np.uint8) * 255
            return pad_and_resize_to_deepprint_input_size(dummy_img, fill=1.0)
            
        # 2. 尝试用 OpenCV 读取图像
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        # 3. 如果 OpenCV 报错或返回 None（防止图像损坏导致崩溃）
        if img is None:
            print(f"\n[警告] 无法读取图像(可能已损坏)，使用纯白占位图跳过: {filepath}")
            dummy_img = np.ones((512, 512), dtype=np.uint8) * 255
            return pad_and_resize_to_deepprint_input_size(dummy_img, fill=1.0)
            
        # 统一填充并缩放到模型所需的输入尺寸
        return pad_and_resize_to_deepprint_input_size(img, fill=1.0)