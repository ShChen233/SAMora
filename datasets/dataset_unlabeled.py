import os
import glob
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset
from einops import repeat


def _is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in {".npy", ".npz", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".h5", ".hdf5"}


def _normalize_to_zero_one(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image = image.astype(np.float32)
    if image.size == 0:
        return image
    min_val = float(np.min(image))
    max_val = float(np.max(image))
    if max_val - min_val < eps:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val + eps)


def _normalize_to_standard(image: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    image = image.astype(np.float32)
    mean = float(np.mean(image))
    std = float(np.std(image))
    if std < eps:
        return image - mean
    return (image - mean) / (std + eps)


def _ensure_hw(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        # Prefer selecting a middle slice if this is a volume.
        smallest_axis = int(np.argmin(image.shape))
        if smallest_axis == 0:
            return image[image.shape[0] // 2]
        if smallest_axis == 1:
            return image[:, image.shape[1] // 2, :]
        return image[:, :, image.shape[2] // 2]
    raise ValueError(f"Expected 2D or 3D array, got shape={image.shape}")


def _read_first_available_key(npz_obj: np.lib.npyio.NpzFile, preferred: Sequence[str]) -> np.ndarray:
    for key in preferred:
        if key in npz_obj:
            return npz_obj[key]
    if len(npz_obj.files) == 0:
        raise ValueError("Empty npz file encountered.")
    return npz_obj[npz_obj.files[0]]


def _load_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path)
        return _ensure_hw(arr)
    if ext == ".npz":
        arr = np.load(path)
        try:
            image = _read_first_available_key(arr, preferred=["image", "img", "data", "volume"])
        finally:
            arr.close()
        return _ensure_hw(image)
    if ext in {".h5", ".hdf5"}:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            for key in ["image", "img", "data", "volume"]:
                if key in f:
                    return _ensure_hw(f[key][:])
            if not keys:
                raise ValueError(f"No dataset found in h5 file: {path}")
            return _ensure_hw(f[keys[0]][:])
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Failed to read image file: {path}")
    return image.astype(np.float32)


def _resize_image(image: np.ndarray, output_size: Tuple[int, int], order: int = 3) -> np.ndarray:
    h, w = image.shape
    out_h, out_w = output_size
    if h == out_h and w == out_w:
        return image.astype(np.float32)
    return zoom(image, (out_h / h, out_w / w), order=order).astype(np.float32)


def _to_three_channel_tensor(image: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
    return repeat(tensor, 'c h w -> (repeat c) h w', repeat=3)


def random_rot_flip(image: np.ndarray) -> np.ndarray:
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    return image


def random_rotate(image: np.ndarray, angle_range: Tuple[int, int] = (-20, 20)) -> np.ndarray:
    angle = np.random.randint(angle_range[0], angle_range[1] + 1)
    return ndimage.rotate(image, angle, order=1, reshape=False)


def random_scale_crop(image: np.ndarray, min_ratio: float = 0.6, max_ratio: float = 1.0) -> np.ndarray:
    h, w = image.shape
    scale = min_ratio + random.random() * (max_ratio - min_ratio)
    new_h = max(1, int(h * scale))
    new_w = max(1, int(w * scale))
    if new_h >= h or new_w >= w:
        return image
    y = np.random.randint(0, h - new_h + 1)
    x = np.random.randint(0, w - new_w + 1)
    return image[y:y + new_h, x:x + new_w]


def random_brightness_contrast(
    image: np.ndarray,
    brightness_delta: float = 0.15,
    contrast_range: Tuple[float, float] = (0.85, 1.15),
) -> np.ndarray:
    alpha = np.random.uniform(contrast_range[0], contrast_range[1])
    beta = np.random.uniform(-brightness_delta, brightness_delta)
    out = image * alpha + beta
    return np.clip(out, 0.0, 1.0)


def add_gaussian_noise(image: np.ndarray, sigma: float = 0.08) -> np.ndarray:
    noise = np.random.normal(0.0, sigma, image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


def random_gaussian_blur(image: np.ndarray, p: float = 0.3) -> np.ndarray:
    if random.random() > p:
        return image
    ksize = random.choice([3, 5])
    sigma = random.uniform(0.6, 1.4)
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)


def build_image_level_views(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    view1 = image.copy()
    view2 = image.copy()

    if random.random() > 0.5:
        view1 = random_rot_flip(view1)
    if random.random() > 0.5:
        view2 = random_rot_flip(view2)

    if random.random() > 0.4:
        view1 = random_rotate(view1)
    if random.random() > 0.4:
        view2 = random_rotate(view2)

    if random.random() > 0.5:
        view1 = random_scale_crop(view1)
    if random.random() > 0.5:
        view2 = random_scale_crop(view2)

    view1 = _normalize_to_zero_one(view1)
    view2 = _normalize_to_zero_one(view2)
    view1 = random_brightness_contrast(view1)
    view2 = random_brightness_contrast(view2)
    view1 = random_gaussian_blur(view1)
    view2 = random_gaussian_blur(view2)
    return view1, view2


class UnlabeledRandomGenerator(object):
    def __init__(self, output_size: Tuple[int, int], ssl_task: str = "image") -> None:
        self.output_size = output_size
        self.ssl_task = ssl_task

    def __call__(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        image = _ensure_hw(image)
        image = _normalize_to_zero_one(image)

        if self.ssl_task == "image":
            view1, view2 = build_image_level_views(image)
            view1 = _resize_image(view1, self.output_size, order=3)
            view2 = _resize_image(view2, self.output_size, order=3)
            clean = _resize_image(image, self.output_size, order=3)
            return {
                "view1": _to_three_channel_tensor(_normalize_to_standard(view1)),
                "view2": _to_three_channel_tensor(_normalize_to_standard(view2)),
                "image": _to_three_channel_tensor(_normalize_to_standard(clean)),
            }

        if self.ssl_task == "patch":
            resized = _resize_image(image, self.output_size, order=3)
            standardized = _normalize_to_standard(resized)
            return {
                "image": _to_three_channel_tensor(standardized),
                # MAE-style masking is typically created inside the model/teacher.
                # Keep the dataloader simple and deterministic.
                "mask": torch.zeros(self.output_size, dtype=torch.float32),
            }

        if self.ssl_task == "pixel":
            clean = _resize_image(image, self.output_size, order=3)
            noisy = add_gaussian_noise(clean)
            if random.random() > 0.5:
                noisy = random_gaussian_blur(noisy, p=1.0)
            return {
                "image": _to_three_channel_tensor(_normalize_to_standard(clean)),
                "noisy_image": _to_three_channel_tensor(_normalize_to_standard(noisy)),
            }

        raise ValueError(f"Unsupported ssl_task: {self.ssl_task}")


class UnlabeledMedicalSliceDataset(Dataset):
    """
    Generic unlabeled medical slice dataset for SAMora stage-1 pretraining.

    Expected usage:
        dataset = UnlabeledMedicalSliceDataset(
            root_dirs=["/path/to/amos", "/path/to/lits"],
            image_size=224,
            ssl_task="image",
        )

    Supported source formats:
    - .npy
    - .npz  (prefers keys: image/img/data/volume)
    - .h5/.hdf5 (prefers keys: image/img/data/volume)
    - common image files (.png/.jpg/.tif/...)
    """

    def __init__(
        self,
        root_dirs: Sequence[str],
        image_size: int = 224,
        ssl_task: str = "image",
        file_patterns: Optional[Sequence[str]] = None,
        transform: Optional[Callable[[np.ndarray], Dict[str, torch.Tensor]]] = None,
        max_samples: Optional[int] = None,
        recursive: bool = True,
        require_nonempty: bool = True,
    ) -> None:
        super().__init__()
        if ssl_task not in {"image", "patch", "pixel"}:
            raise ValueError("ssl_task must be one of {'image', 'patch', 'pixel'}")
        self.ssl_task = ssl_task
        self.image_size = image_size
        self.output_size = (image_size, image_size)
        self.transform = transform or UnlabeledRandomGenerator(self.output_size, ssl_task=ssl_task)
        self.file_patterns = list(file_patterns) if file_patterns is not None else [
            "*.npy", "*.npz", "*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.h5", "*.hdf5"
        ]
        self.recursive = recursive

        self.sample_paths = self._discover_files(root_dirs)
        if max_samples is not None and len(self.sample_paths) > max_samples:
            self.sample_paths = self.sample_paths[:max_samples]

        if require_nonempty and len(self.sample_paths) == 0:
            raise ValueError(
                "No unlabeled image files were found. Check root_dirs / file_patterns for stage-1 pretraining."
            )

    def _discover_files(self, root_dirs: Sequence[str]) -> List[str]:
        results: List[str] = []
        for root_dir in root_dirs:
            if not os.path.exists(root_dir):
                continue
            for pattern in self.file_patterns:
                if self.recursive:
                    matched = glob.glob(os.path.join(root_dir, "**", pattern), recursive=True)
                else:
                    matched = glob.glob(os.path.join(root_dir, pattern), recursive=False)
                results.extend(matched)
        results = [p for p in results if os.path.isfile(p) and _is_image_file(p)]
        results = sorted(set(results))
        return results

    def __len__(self) -> int:
        return len(self.sample_paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.sample_paths[idx]
        image = _load_image(path)
        sample = self.transform(image)
        sample["path"] = path
        return sample
