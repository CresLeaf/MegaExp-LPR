from pathlib import Path
from typing import List, Union

VALID_IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]


def get_image_files(path: Union[str, Path]) -> List[Path]:
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
        if input_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image format: {input_path.suffix}")
        return [input_path]
    elif input_path.is_dir():
        image_files = [
            f
            for f in input_path.glob("**/*")
            if f.suffix.lower() in VALID_IMAGE_EXTENSIONS
        ]
        if not image_files:
            raise ValueError(f"No valid image files found in directory: {path}")
        return image_files
    else:
        raise ValueError(f"Invalid path: {path}")
