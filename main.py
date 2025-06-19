import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path

from detection import LicensePlateDetector
from recognition import RecognitionModel
from training import train_license_plate_model


def create_recognition_dataset(data_dir: str, batch_size: int = 32):
    """Create training and validation datasets for recognition model"""
    # You'll need to implement a proper dataset class that loads segmented license plates
    # and their corresponding text labels

    train_dir = Path(data_dir) / "segments" / "train"
    val_dir = Path(data_dir) / "segments" / "val"

    # For now, using the SegmentLoader as a base
    # You'll need to extend this to include text labels
    train_dataset = SegmentLoader(str(train_dir))
    val_dataset = SegmentLoader(str(val_dir))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_recognition_model(args):
    """Train the license plate recognition model"""
    print("Starting license plate recognition training...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters
    input_size = 3  # RGB channels
    hidden_size = 256
    # Output size should match your character vocabulary
    # Based on recognition.py, this includes provinces + letters + digits + special chars
    output_size = 50  # Adjust based on your actual vocabulary size

    # Create model
    model = RecognitionModel(
        input_size=input_size, hidden_size=hidden_size, output_size=output_size
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Create data loaders
    try:
        train_loader, val_loader = create_recognition_dataset(
            args.data_dir, batch_size=args.batch_size
        )
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
    except Exception as e:
        print(f"Error creating datasets: {e}")
        print("Please ensure your data directory structure is correct:")
        print("  data_dir/train/segmented_plates/")
        print("  data_dir/val/segmented_plates/")
        return None

    # Train the model
    trained_model, history = train_license_plate_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        save_dir=args.save_dir,
        early_stopping_patience=args.patience,
        device=device,
    )

    print("Training completed!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Model saved to: {args.save_dir}")

    return trained_model, history


def main():
    dataset_desc = "datasets/CRPD_split"

    detector = LicensePlateDetector(model_path="yolo_model.pt")

    detections = detector.detect_batch(image_path=dataset_desc)

    detector.segment(detections)

    train_loader, val_loader = create_recognition_dataset(
        dataset_desc,
        batch_size=32,
    )
