import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
import os

from recognition import default_vocab

to_tensor = transforms.ToTensor()


class IndexedDataset(Dataset):
    def __init__(
        self,
        csv_path,
        root_dir="",
        transform=None,
        encoding="gb2312",
        vocab=default_vocab,
    ):
        """
        Args:
            csv_path: Path to CSV file
            root_dir: Prefix path for image files
            transform: Optional transform to be applied on an image
            encoding: Encoding of the CSV file (e.g., 'gb2312' for Chinese)
            vocab: List of characters for recognition (if None, build from CSV)
        """
        self.data = pd.read_csv(csv_path, encoding=encoding)
        self.root_dir = root_dir
        self.transform = transform

        # Handle vocabulary
        if vocab is None:
            all_labels = "".join(self.data["label"].astype(str))
            self.vocab = sorted(set(all_labels))
        else:
            self.vocab = vocab

        self.char2idx = {char: i for i, char in enumerate(self.vocab)}
        self.idx2char = {i: char for char, i in self.char2idx.items()}

    def __len__(self):
        return len(self.data)

    def encode_label(self, label):
        return torch.tensor([self.char2idx[c] for c in label], dtype=torch.long)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image
        img_path = os.path.join(self.root_dir, row["path"])
        image = Image.open(img_path).convert("RGB")
        image = to_tensor(image)

        # Get polygon (4 points)
        coords = torch.tensor(
            [
                [row["x1"], row["y1"]],
                [row["x2"], row["y2"]],
                [row["x3"], row["y3"]],
                [row["x4"], row["y4"]],
            ],
            dtype=torch.float32,
        )

        # Character recognition label
        label_str = row["label"]
        label_encoded = self.encode_label(label_str)

        if self.transform:
            image = self.transform(image, coords)

        return {
            "image": image,
            "polygon": coords,  # segmentation/localization
            "label": label_encoded,  # recognition
            "label_str": label_str,  # optionally keep raw string
        }


class SpareDataset(Dataset):
    def __init__(self, root_dir, transform=None, vocab=default_vocab, encoding="utf-8"):
        """
        Args:
            root_dir: Path to 'train/', 'val/', or 'test/' directory
            transform: Optional torchvision transform
            vocab: List of characters for encoding labels
        """
        self.image_dir = os.path.join(root_dir, "images")
        self.label_dir = os.path.join(root_dir, "labels")
        self.transform = transform

        self.image_paths = sorted(
            [
                os.path.join(self.image_dir, fname)
                for fname in os.listdir(self.image_dir)
                if fname.lower().endswith((".jpg", ".png", ".jpeg"))
            ]
        )

        if vocab is None:
            all_chars = set()
            for img_path in self.image_paths:
                label_path = self._get_label_path(img_path)
                with open(label_path, encoding=encoding) as f:
                    label = f.readline().strip()
                    all_chars.update(label)
            self.vocab = sorted(all_chars)
        else:
            self.vocab = vocab

        self.char2idx = {char: i for i, char in enumerate(self.vocab)}
        self.idx2char = {i: char for char, i in self.char2idx.items()}

    def _get_label_path(self, image_path):
        filename = os.path.basename(image_path)
        name, _ = os.path.splitext(filename)
        return os.path.join(self.label_dir, f"{name}.txt")

    def encode_label(self, label):
        return torch.tensor([self.char2idx[c] for c in label], dtype=torch.long)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self._get_label_path(img_path)

        image = Image.open(img_path).convert("RGB")
        image = to_tensor(image)

        with open(label_path, encoding="utf-8") as f:
            raw = f.readline().split()
            coords = [float(coord) for coord in raw[:8]]
            polygon = torch.tensor(coords[:8], dtype=torch.float32).view(4, 2)
            label_str = raw[9]

        label_encoded = self.encode_label(label_str)

        if self.transform:
            image = self.transform(image, polygon)

        return {
            "image": image,
            "label": label_encoded,
            "label_str": label_str,
            "image_path": img_path,
            "polygon": polygon,
        }


def labelled_segment(image, polygon):
    """
    Segment the images for recognition training with known polygon coordinates.
    Performs slicing (this breaks batch since each image may have different sizes).
    """
    # polygon already absolute coordinates
    x_min = int(polygon[:, 0].min().item())
    x_max = int(polygon[:, 0].max().item())
    y_min = int(polygon[:, 1].min().item())
    y_max = int(polygon[:, 1].max().item())

    # Crop the image and polygon
    image = image[:, y_min:y_max, x_min:x_max]

    return image


def test():
    dataset = IndexedDataset(
        csv_path="datasets/CLPD/CLPD.csv",
        root_dir="datasets/CLPD/",
        transform=labelled_segment,
    )

    loader = torch.utils.data.DataLoader(dataset, batch_size=1)

    for i, batch in enumerate(loader):
        print("Batch size:", len(batch["image"]))
        print("Image shape:", batch["image"].shape)
        print("Polygon shape:", batch["polygon"].shape)
        print("Label shape:", batch["label"].shape)
        print("Label string:", batch["label_str"])
        if i >= 10:
            break

    dataset = SpareDataset(
        root_dir="datasets/val",
        transform=labelled_segment,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=1)
    for i, batch in enumerate(loader):
        print("Batch size:", len(batch["image"]))
        print("Image shape:", batch["image"].shape)
        print("Polygon shape:", batch["polygon"].shape)
        print("Label shape:", batch["label"].shape)
        print("Label string:", batch["label_str"])
        if i >= 10:
            break


if __name__ == "__main__":
    test()
