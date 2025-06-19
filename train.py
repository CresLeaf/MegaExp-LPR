"""
Old labels:
    1st: To figure out
    2nd: 0~23 for alphabet, except O and I
    remaining: 0~23 for alphabet, except O and I, 24~33 for digits
New labels:
    the idxs of the characters in the <blank>, LF, digits, alphabet(no O and I) order.
Converts old license plate labels to new format.
"""

from itertools import chain
import sys

province_mapping = {
    0: 16,  # 皖
    1: 9,  # 沪
    19: 17,  # 粤
    22: 5,  # 川
    10: 18,  # 苏
    14: 23,  # 赣
    11: 11,  # 浙
    7: 24,  # 辽
    12: 1,  # 京
    15: 29,  # 鲁
    17: 25,  # 鄂
    13: 26,  # 闽
    16: 9999,  # ???
    4: 2,  # 冀
    18: 13,  # 湘
    9: 30,  # 黑
    26: 27,  # 陕
    6: 19,  # 蒙
}
province_mapping = dict(
    ((k, v + 2 + 10 + 24) for k, v in province_mapping.items())
)  # convert to new idxs
mapping = dict(
    chain(
        ((x, x - 21) for x in range(24, 34)),  # digits 1~9
        ((x, x + 12) for x in range(0, 24)),  # alphabet A~Z except O and I
    )
)


def convert_labels(label: str) -> str:
    # split the path by dash
    old = label.split("-")[4]
    # parts = old.split("_")
    # try:
    #     new = list(
    #         chain(
    #             [province_mapping[int(parts[0])]],
    #             [mapping[int(x)] for x in parts[1:]],
    #         )
    #     )  # Trigger error to figure out the province mapping
    #     return "_".join(map(str, new))
    # except KeyError:
    #     print(f"missing mapping for {old}, please update province_mapping")
    #     sys.exit(1)
    return old


def segment_train(description: str) -> None:
    """
    Read the effing dataset description and segment the training and validation sets.

    Args:
        description (str): Path to the dataset description YAML file.

    Effects:
        Segments the dataset under dataset root, ignores the fucking upstream who mixes configuration and convention.

    Note:
        This function assumes the dataset is structured as follows:
        - `data_dir/{images, labels}/{train, val}/`
        which is awfully documented in the original repo.
        Produce the segmented dataset under `data_dir/{train, val}/{segments, labels}`,
        so it can be consistent with prediction and evaluation scripts.
        train boxes use relative coordinates, must be converted to absolute coordinates.
    """
    from pathlib import Path
    import cv2
    import yaml

    import utils

    with open(description, "r") as f:
        config = yaml.safe_load(f)

    #
    data_dir = Path(config["path"])
    train_images = data_dir / "images" / "train"
    train_boxes = data_dir / "labels" / "train"
    val_images = data_dir / "images" / "val"
    val_boxes = data_dir / "labels" / "val"

    # Create new directories for segmented dataset
    train_segments = data_dir / "segments" / "train"
    val_segments = data_dir / "segments" / "val"
    train_segments.mkdir(parents=True, exist_ok=True)
    val_segments.mkdir(parents=True, exist_ok=True)

    # process training images
    for img_path in train_images.glob("*"):
        if img_path.suffix.lower() in utils.VALID_IMAGE_EXTENSIONS:
            # fetch the corresponding label file
            label_path = train_boxes / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"Warning: No label file for {img_path}, skipping.")
                continue

            # read the image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Error reading image {img_path}, skipping.")
                continue

            # iterate over each line in the label file
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        print(f"Invalid label format in {label_path}, skipping line.")
                        continue

                    # parse the label
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # convert to absolute coordinates
                    x1 = int((x_center - width / 2) * img.shape[1])
                    y1 = int((y_center - height / 2) * img.shape[0])
                    x2 = int((x_center + width / 2) * img.shape[1])
                    y2 = int((y_center + height / 2) * img.shape[0])

                    # segment the image
                    segment = img[y1:y2, x1:x2]
                    if segment.size == 0:
                        print(f"Empty segment for {img_path} at {x1}, {y1}, {x2}, {y2}")
                        continue

                    # save the segmented image
                    segment_filename = (
                        train_segments / convert_labels(img_path.stem)
                    ).with_suffix(".png")
                    cv2.imwrite(str(segment_filename), segment)

    # process validation images
    for img_path in val_images.glob("*"):
        if img_path.suffix.lower() in utils.VALID_IMAGE_EXTENSIONS:
            # fetch the corresponding label file
            label_path = val_boxes / (img_path.stem + ".txt")
            if not label_path.exists():
                print(f"Warning: No label file for {img_path}, skipping.")
                continue

            # read the image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Error reading image {img_path}, skipping.")
                continue

            # iterate over each line in the label file
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        print(f"Invalid label format in {label_path}, skipping line.")
                        continue

                    # parse the label
                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # convert to absolute coordinates
                    x1 = int((x_center - width / 2) * img.shape[1])
                    y1 = int((y_center - height / 2) * img.shape[0])
                    x2 = int((x_center + width / 2) * img.shape[1])
                    y2 = int((y_center + height / 2) * img.shape[0])

                    # segment the image
                    segment = img[y1:y2, x1:x2]
                    if segment.size == 0:
                        print(f"Empty segment for {img_path} at {x1}, {y1}, {x2}, {y2}")
                        continue

                    # save the segmented image
                    segment_filename = (
                        val_segments / convert_labels(img_path.stem)
                    ).with_suffix(".png")
                    cv2.imwrite(str(segment_filename), segment)


import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from typing import Tuple, List, Optional


class LicensePlateSegmentDataset(Dataset):
    """Dataset for license plate character segments"""

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
    ):
        """
        Args:
            root_dir: Root directory containing segments folder
            split: 'train' or 'val'
            transform: Image transformations to apply
        """
        self.root_dir = root_dir
        self.split = split
        self.data_dir = os.path.join(root_dir, "segments", split)
        self.transform = transform

        # Load image paths and extract targets from filenames
        self.samples = self._load_samples()

        if not self.samples:
            raise ValueError(f"No samples found in {self.data_dir}")

    def _load_samples(self) -> List[Tuple[str, List[int]]]:
        """Load image paths and extract targets from filenames"""
        samples = []

        if not os.path.exists(self.data_dir):
            raise ValueError(f"Directory does not exist: {self.data_dir}")

        for filename in os.listdir(self.data_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # Extract target from filename (remove extension)
            target_str = os.path.splitext(filename)[0]

            # Parse the target string - assuming format like "0_1_32_3_3"
            # Each number/character separated by underscore
            target_chars = map(int, target_str.split("_"))

            if target_chars:  # Only add if we successfully parsed the target
                image_path = os.path.join(self.data_dir, filename)
                samples.append((image_path, target_chars))

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Returns:
            image: Transformed image tensor
            target: Target sequence as tensor
            target_length: Length of target sequence
        """
        image_path, target_indices = self.samples[idx]

        # Load image
        image = Image.open(image_path).convert("RGB")

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Convert target to tensor
        target = torch.tensor(target_indices, dtype=torch.long)
        target_length = len(target_indices)

        return image, target, target_length

    def get_char_mapping(self) -> Tuple[dict, dict]:
        """Return character mappings"""
        return None, None


def collate_fn(batch) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Custom collate function for variable length sequences"""
    images, targets, target_lengths = zip(*batch)

    # Stack images
    images = torch.stack(images, dim=0)

    # Handle variable length targets
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)

    # Concatenate all targets into a single tensor (CTC requirement)
    targets = torch.cat(targets, dim=0)

    return images, targets, target_lengths


def get_default_transforms(image_size: Tuple[int, int] = (32, 128)) -> dict:
    """Get default image transformations for training and validation"""

    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            ),
            transforms.RandomAffine(
                degrees=2, translate=(0.05, 0.05), scale=(0.95, 1.05)
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return {"train": train_transform, "val": val_transform}


def create_dataloaders(
    root_dir: str,
    batch_size: int = 32,
    image_size: Tuple[int, int] = (32, 128),
    num_workers: int = 4,
    char_to_idx: Optional[dict] = None,
) -> Tuple[DataLoader, DataLoader, dict, dict]:
    """
    Create train and validation dataloaders

    Returns:
        train_loader, val_loader, char_to_idx, idx_to_char
    """

    # Get transforms
    transforms_dict = get_default_transforms(image_size)

    # Create datasets
    train_dataset = LicensePlateSegmentDataset(
        root_dir=root_dir,
        split="train",
        transform=transforms_dict["train"],
        # char_to_idx=char_to_idx,
    )

    val_dataset = LicensePlateSegmentDataset(
        root_dir=root_dir,
        split="val",
        transform=transforms_dict["val"],
        # char_to_idx=train_dataset.char_to_idx,  # Use same mapping
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # char_to_idx, idx_to_char = train_dataset.get_char_mapping()
    char_to_idx, idx_to_char = {}, {}

    print(f"Dataset loaded:")
    print(f"  Training samples: {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    # print(f"  Character vocabulary size: {len(char_to_idx)}")
    print(f"  Batch size: {batch_size}")

    return train_loader, val_loader, char_to_idx, idx_to_char


# Example usage function
def load_license_plate_data(root_dir: str, batch_size: int = 32):
    """
    Convenience function to load license plate segment data

    Args:
        root_dir: Path to directory containing segments/{train,val}/ folders
        batch_size: Batch size for training

    Returns:
        train_loader, val_loader, char_to_idx, idx_to_char
    """
    return create_dataloaders(root_dir, batch_size=batch_size)


if __name__ == "__main__":
    # Example usage
    root_dir = "datasets/CRPD_split"  # Adjust path as needed

    try:
        train_loader, val_loader, char_to_idx, idx_to_char = load_license_plate_data(
            root_dir, batch_size=16
        )

        # Test loading a batch
        for images, targets, target_lengths in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Targets shape: {targets.shape}")
            print(f"Target lengths: {target_lengths}")

            # Decode first sample
            start_idx = 0
            end_idx = target_lengths[0].item()
            first_target = targets[start_idx:end_idx]
            # decoded = "".join([idx_to_char[idx.item()] for idx in first_target])
            print(f"First sample target: {first_target.tolist()}")
            break

    except Exception as e:
        print(f"Error loading data: {e}")
