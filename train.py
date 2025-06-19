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


def main():
    description = "crpd.yml"
    segment_train(description)
    print("Dataset segmentation completed.")


if __name__ == "__main__":
    main()
