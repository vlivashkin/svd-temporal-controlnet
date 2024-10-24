import json
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

CONTEXT_LENGTH = 14


def pil_image_to_numpy(image):
    """Convert a PIL image to a NumPy array."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.array(image)


def numpy_to_pt(images: np.ndarray) -> torch.FloatTensor:
    """Convert a NumPy image to a PyTorch tensor."""
    if images.ndim == 3:
        images = images[..., None]
    images = torch.from_numpy(images.transpose(0, 3, 1, 2))
    return images.float() / 255


class WebVid10M(Dataset):
    def __init__(self, csv_path, video_folder, depth_folder, _, sample_size=256, sample_stride=4, sample_n_frames=14):
        with open(csv_path) as f:
            self.dataset = json.load(f)
        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        random.shuffle(self.dataset)
        self.video_folder = video_folder
        self.sample_stride = sample_stride
        self.sample_n_frames = sample_n_frames
        self.depth_folder = depth_folder
        print("length", len(self.dataset))
        sample_size = tuple(sample_size) if not isinstance(sample_size, int) else (sample_size, sample_size)
        print("sample size", sample_size)
        self.pixel_transforms = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.Resize(sample_size),
                transforms.CenterCrop(sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

    def center_crop(self, img):
        h, w = img.shape[-2:]  # Assuming img shape is [C, H, W] or [B, C, H, W]
        min_dim = min(h, w)
        top = (h - min_dim) // 2
        left = (w - min_dim) // 2
        return img[..., top : top + min_dim, left : left + min_dim]

    def get_batch(self, idx):
        def sort_frames(frame_name):
            return int(frame_name.split("_")[1].split(".")[0])

        while True:
            video_dict = self.dataset[idx]

            start_frame_idx = np.random.randint(low=0, high=len(video_dict["frames"]) - CONTEXT_LENGTH)
            image_files, depth_files = [], []
            for idx in range(CONTEXT_LENGTH):
                frame = video_dict["frames"][start_frame_idx + idx]
                image_files.append(frame["img_path"])
                depth_files.append(frame["cond_img_path"])

            # Load image frames
            numpy_images = np.array([pil_image_to_numpy(Image.open(img)) for img in image_files])
            pixel_values = numpy_to_pt(numpy_images)

            # Load depth frames
            numpy_depth_images = np.array([pil_image_to_numpy(Image.open(df)) for df in depth_files])
            depth_pixel_values = numpy_to_pt(numpy_depth_images)

            # Load motion values
            motion_values = 0.0

            return pixel_values, depth_pixel_values, motion_values

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        pixel_values, depth_pixel_values, motion_values = self.get_batch(idx)
        pixel_values = self.pixel_transforms(pixel_values)
        depth_pixel_values = depth_pixel_values[:, :, ::2, ::2]  # FIXME: for 256x256
        sample = dict(pixel_values=pixel_values, depth_pixel_values=depth_pixel_values, motion_values=motion_values)
        return sample
