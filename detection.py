"""
License Plate Detection Library

A modular library for detecting license plates using YOLO models.
Supports both offline batch processing and single image detection.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple
from tqdm import tqdm

import torch
from ultralytics import YOLO


class LicensePlateDetector:
    """
    License plate detector using YOLO models.
    """

    VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

    def __init__(self, model_path: str):
        """
        Initialize the detector with a trained YOLO model.

        Args:
            model_path (str): Path to the trained YOLO model file
        """
        self.model = YOLO(model_path)
        self.model_path = model_path

    def _get_image_files(self, path: Union[str, Path]) -> List[Path]:
        """
        Get list of image files from path (file or directory).

        Args:
            path: Path to image file or directory containing images

        Returns:
            List of Path objects for valid image files

        Raises:
            ValueError: If path is invalid or no images found
        """
        input_path = Path(path)

        if input_path.is_file():
            if input_path.suffix.lower() not in self.VALID_IMAGE_EXTENSIONS:
                raise ValueError(f"Unsupported image format: {input_path.suffix}")
            return [input_path]
        elif input_path.is_dir():
            image_files = [
                f
                for f in input_path.glob("**/*")
                if f.suffix.lower() in self.VALID_IMAGE_EXTENSIONS
            ]
            if not image_files:
                raise ValueError(f"No valid image files found in directory: {path}")
            return image_files
        else:
            raise ValueError(f"Invalid path: {path}")

    def detect_single_image(
        self, image_path: Union[str, Path], conf_threshold: float = 0.25
    ) -> List[Dict]:
        """
        Detect license plates in a single image.

        Args:
            image_path: Path to image file
            conf_threshold: Confidence threshold for detections

        Returns:
            List of detection dictionaries containing bbox, confidence, class info
        """
        results = self.model(image_path, conf=conf_threshold)

        detections = []
        if results and len(results) > 0:
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract bounding box coordinates
                        xyxy = box.xyxy.cpu().numpy().tolist()[0]

                        # Extract confidence and class information
                        conf = float(box.conf.cpu().numpy()[0])
                        cls = int(box.cls.cpu().numpy()[0])
                        cls_name = result.names.get(cls, str(cls))

                        detections.append(
                            {
                                "bbox": xyxy,  # [x1, y1, x2, y2]
                                "confidence": conf,
                                "class_id": cls,
                                "class_name": cls_name,
                            }
                        )

        return detections

    def detect_batch(
        self,
        image_path: Union[str, Path],
        conf_threshold: float = 0.25,
        show_progress: bool = True,
    ) -> Dict[str, List[Dict]]:
        """
        Detect license plates in multiple images.

        Args:
            image_path: Path to image file or directory
            conf_threshold: Confidence threshold for detections
            show_progress: Whether to show progress bar

        Returns:
            Dictionary mapping image paths to detection results
        """
        image_files = self._get_image_files(image_path)
        results_dict = {}

        iterator = (
            tqdm(image_files, desc="Processing images")
            if show_progress
            else image_files
        )

        for img_file in iterator:
            detections = self.detect_single_image(img_file, conf_threshold)
            results_dict[str(img_file)] = detections

        return results_dict

    def save_results(
        self,
        results: Dict[str, List[Dict]],
        output_path: Union[str, Path],
        separate_files: bool = False,
    ) -> None:
        """
        Save detection results to JSON file(s).

        Args:
            results: Detection results dictionary
            output_path: Path for output JSON file
            separate_files: Whether to save separate JSON for each image
        """
        output_path = Path(output_path)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if separate_files:
            output_dir = output_path.parent
            for img_path, detections in results.items():
                img_name = Path(img_path).stem
                json_file = output_dir / f"{img_name}_detection.json"

                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump({img_path: detections}, f, ensure_ascii=False, indent=2)

        # Always save combined results
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    def visualize_results(
        self,
        image_path: Union[str, Path],
        output_dir: Union[str, Path],
        conf_threshold: float = 0.25,
        show_progress: bool = True,
    ) -> None:
        """
        Generate visualization of detection results.

        Args:
            image_path: Path to image file or directory
            output_dir: Directory to save visualized results
            conf_threshold: Confidence threshold for detections
            show_progress: Whether to show progress bar
        """
        image_files = self._get_image_files(image_path)
        output_dir = Path(output_dir)

        # Create visualization subdirectory
        vis_dir = output_dir / "visualized"
        vis_dir.mkdir(parents=True, exist_ok=True)

        iterator = (
            tqdm(image_files, desc="Generating visualizations")
            if show_progress
            else image_files
        )

        for img_file in iterator:
            # Use YOLO's built-in visualization
            self.model(
                img_file,
                conf=conf_threshold,
                save=True,
                project=str(vis_dir),
                name="",
                exist_ok=True,
            )


class DetectionConfig:
    """Configuration class for detection parameters."""

    def __init__(
        self,
        conf_threshold: float = 0.25,
        save_separate_json: bool = False,
        enable_visualization: bool = False,
        show_progress: bool = True,
    ):
        self.conf_threshold = conf_threshold
        self.save_separate_json = save_separate_json
        self.enable_visualization = enable_visualization
        self.show_progress = show_progress

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            "conf_threshold": self.conf_threshold,
            "save_separate_json": self.save_separate_json,
            "enable_visualization": self.enable_visualization,
            "show_progress": self.show_progress,
        }


def detect_license_plates(
    model_path: str,
    image_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    config: Optional[DetectionConfig] = None,
) -> Dict[str, List[Dict]]:
    """
    Convenience function for license plate detection.

    Args:
        model_path: Path to trained YOLO model
        image_path: Path to image(s) for detection
        output_path: Path to save results (optional)
        config: Detection configuration (optional)

    Returns:
        Dictionary containing detection results
    """
    if config is None:
        config = DetectionConfig()

    detector = LicensePlateDetector(model_path)
    results = detector.detect_batch(
        image_path,
        conf_threshold=config.conf_threshold,
        show_progress=config.show_progress,
    )

    if output_path:
        detector.save_results(
            results, output_path, separate_files=config.save_separate_json
        )

    if config.enable_visualization and output_path:
        vis_output = Path(output_path).parent
        detector.visualize_results(
            image_path,
            vis_output,
            conf_threshold=config.conf_threshold,
            show_progress=config.show_progress,
        )

    return results


# Backward compatibility functions
def detect_single_plate(
    model_path: str, image_path: str, conf_threshold: float = 0.25
) -> List[Dict]:
    """Detect license plates in a single image (backward compatibility)."""
    detector = LicensePlateDetector(model_path)
    return detector.detect_single_image(image_path, conf_threshold)


def batch_detect_plates(
    model_path: str, images_dir: str, conf_threshold: float = 0.25
) -> Dict[str, List[Dict]]:
    """Detect license plates in multiple images (backward compatibility)."""
    detector = LicensePlateDetector(model_path)
    return detector.detect_batch(images_dir, conf_threshold)
