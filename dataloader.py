import cv2
import os
import torch
from typing import Dict, Any, List
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SegmentConfig:
    root_dir: str
    target_dir: str = "segments"


def segment(config: SegmentConfig, detections: Dict[str, dict]) -> None:
    """
    Segment the detections into individual license plate images.

    Args:
        detections (Dict[str, Any]): Detection indexed by image paths.

    Effects:
        Save segmented license plate images to disk.
    """
    for img_path, detection in detections.items():
        img = cv2.imread(img_path)
        path = Path(img_path)
        if img is None:
            print(f"Error reading image {img_path}")
            continue
        for i, box in enumerate(detection.get("boxes", [])):
            x1, y1, x2, y2 = box["bbox"]
            # Ensure coordinates are within image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            segmented_plate = img[y1:y2, x1:x2]
            if segmented_plate.size == 0:
                print(f"Empty segment for {img_path} at index {i}")
                continue
            # Save the segmented plate image
            output_path = Path(config.root_dir) / config.target_dir / path.name
            os.makedirs(output_path.parent, exist_ok=True)
            cv2.imwrite(
                output_path.with_stem(f"{path.stem}_plate_{i}"), segmented_plate
            )


class SegmentLoader(torch.utils.data.Dataset):
    """
    Custom dataset loader for segmented license plate images.

    Args:
        root_dir (str): Directory containing segmented plate images.
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, root_dir: str, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = list(self.root_dir.glob("*.jpg"))  # Adjust as needed

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path = self.image_paths[idx]
        image = cv2.imread(str(img_path))
        if self.transform:
            image = self.transform(image)
        return {"image": image, "path": img_path}
