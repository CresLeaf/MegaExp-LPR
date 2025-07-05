import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from pathlib import Path

from dataloader import IndexedDataset, SpareDataset, labelled_segment
from detection import LicensePlateDetector
from recognition import Reco, default_vocab
from training import train_license_plate_model

default_args = {
    "batch_size": 1,
    "num_workers": 4,
    "learning_rate": 0.001,
    "weight_decay": 0.0001,
    "epochs": 50,
    "save_dir": "models/recognition",
    "patience": 5,
    "model": None,
    "train_dataset": None,
    "val_dataset": None,
}


def prepare(args):
    val_dataset = IndexedDataset(
        csv_path="datasets/CLPD/CLPD.csv",
        root_dir="datasets/CLPD/",
        transform=labelled_segment,
    )

    train_dataset = SpareDataset(
        root_dir="datasets/train",
        transform=labelled_segment,
    )

    model = Reco(
        input_size=3,  # Assuming RGB images
        hidden_size=64,
        step_width=64,
        output_size=len(default_vocab),  # Size of the vocabulary
    )

    args["train_dataset"] = train_dataset
    args["val_dataset"] = val_dataset
    args["model"] = model

    return args


def train_recognition_model(args):
    """Train the license plate recognition model"""
    print("Starting license plate recognition training...")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader = DataLoader(
        args["train_dataset"],
        batch_size=args["batch_size"],
        shuffle=True,
        num_workers=args["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        args["val_dataset"],
        batch_size=args["batch_size"],
        shuffle=False,
        num_workers=args["num_workers"],
        pin_memory=True,
    )
    # Train the model
    trained_model, history = train_license_plate_model(
        model=args["model"],
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args["learning_rate"],
        weight_decay=args["weight_decay"],
        epochs=args["epochs"],
        save_dir=args["save_dir"],
        early_stopping_patience=args["patience"],
        device=device,
    )

    print("Training completed!")
    print(f"Best validation loss: {min(history['val_loss']):.4f}")
    print(f"Model saved to: {args.save_dir}")

    return trained_model, history


def test_train():
    args = prepare(default_args)
    trained_model, history = train_recognition_model(args)


if __name__ == "__main__":
    test_train()
