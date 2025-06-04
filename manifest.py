import os
import json
import argparse
import random
from PIL import Image


def parse_coordinates(filename):
    """Extract coordinates from filename format like:
    01-0_0-276&527_420&585-420&583_276&585_276&529_420&527-0_0_7_27_30_30_23-98-6.jpg
    """
    parts = filename.split("-")
    if len(parts) < 3:
        raise ValueError(f"Filename '{filename}' doesn't follow the expected format")

    # Extract coordinates from the 2nd and 3rd parts
    coord_parts = parts[2].split("_")
    if len(coord_parts) < 2:
        raise ValueError(
            f"Coordinate section in '{filename}' doesn't follow the expected format"
        )

    top_left = coord_parts[0].split("&")
    bottom_right = coord_parts[1].split("&")

    if len(top_left) != 2 or len(bottom_right) != 2:
        raise ValueError(
            f"Coordinate values in '{filename}' are not in the expected format"
        )

    x1, y1 = float(top_left[0]), float(top_left[1])
    x2, y2 = float(bottom_right[0]), float(bottom_right[1])

    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def noise(points, max_noise=0.99):
    """Add subpixel noise to coordinates

    Args:
        points: List of [x, y] coordinates
        max_noise: Maximum noise in pixels (default: 0.99 pixels)

    Returns:
        List of points with added subpixel noise
    """
    noisy_points = []

    for x, y in points:
        # Generate random subpixel noise
        noise_x = random.uniform(-max_noise, max_noise)
        noise_y = random.uniform(-max_noise, max_noise)

        # Apply noise
        new_x = x + noise_x
        new_y = y + noise_y

        noisy_points.append([new_x, new_y])

    return noisy_points


def generate_manifest(
    image_path,
    output_dir=None,
    label="1",
    score=None,
    max_noise=0.99,
):
    try:
        with Image.open(image_path) as img:
            width, height = img.size
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        return None

    filename = os.path.basename(image_path)
    base_filename = os.path.splitext(filename)[0]

    try:
        points = parse_coordinates(base_filename)
    except ValueError as e:
        print(f"Error parsing coordinates: {e}")
        return None

    manifest = {
        "version": "2.5.4",
        "flags": {},
        "shapes": [
            {
                "label": label,
                "score": score,
                "points": noise(points, max_noise),
                "group_id": None,
                "description": None,
                "difficult": False,
                "shape_type": "rectangle",
                "flags": {},
                "attributes": {},
                "kie_linking": [],
            }
        ],
        "imagePath": filename,
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
    }

    if output_dir is None:
        output_dir = os.path.dirname(image_path)
    output_path = os.path.join(output_dir, f"{base_filename}.json")
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # print(f"Generated manifest: {output_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate JSON manifest from image file"
    )
    parser.add_argument("image_path", help="Path to the image file")
    parser.add_argument("--output-dir", help="Output directory for the JSON file")
    parser.add_argument("--label", default="1", help="Label for the bounding box")
    parser.add_argument(
        "--score",
        type=float,
        default=0.8753053545951843,
        help="Confidence score for the detection",
    )
    parser.add_argument(
        "--add-noise", action="store_true", help="Add subpixel noise to coordinates"
    )
    parser.add_argument(
        "--max-noise",
        type=float,
        default=0.99,
        help="Maximum subpixel noise in pixels (default: 0.99)",
    )

    args = parser.parse_args()
    generate_manifest(
        args.image_path,
        args.output_dir,
        args.label,
        args.score,
        args.add_noise,
        args.max_noise,
    )
