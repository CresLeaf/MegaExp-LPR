import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
import time
import os
from typing import Dict, Tuple, Optional, List, Any
import matplotlib.pyplot as plt
import numpy as np


def adamw_setup(model, learning_rate=0.001, weight_decay=0.01, epochs=200):
    optimizer = opt.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    # Cosine annealing schedule works well with AdamW
    scheduler = opt.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=learning_rate / 100
    )

    return optimizer, scheduler


class CTCLossWrapper(nn.Module):
    """Wrapper around CTC loss with handling for batch processing"""

    def __init__(self, blank=0, zero_infinity=True):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=blank, zero_infinity=zero_infinity, reduction="mean"
        )

    def forward(self, logits, targets, input_lengths, target_lengths):
        # logits: [batch, time, num_classes]
        # Expected shape for CTCLoss: [time, batch, num_classes]
        logits = logits.transpose(0, 1)
        return self.ctc_loss(logits, targets, input_lengths, target_lengths)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        device: torch.device,
        save_dir: str = "checkpoints",
        max_epochs: int = 200,
        early_stopping_patience: int = 10,
        grad_clip: float = 5.0,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_dir = save_dir
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.grad_clip = grad_clip

        # Loss function
        self.criterion = CTCLossWrapper(blank=0)

        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

    def train_epoch(self) -> float:
        """Run a single training epoch"""
        self.model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (images, targets, target_lengths) in enumerate(
            self.train_loader
        ):
            # Move to device
            images = images.to(self.device)
            targets = targets.to(self.device)
            target_lengths = target_lengths.to(self.device)

            # Forward pass - split into encoder and decoder steps
            self.optimizer.zero_grad()

            # CNN encoder pass
            features = self.model.cnn_encoder(images)

            # LSTM decoder pass
            logits = self.model.ctc_decoder(features)

            # Calculate loss
            input_lengths = torch.full(
                (logits.size(0),), logits.size(1), dtype=torch.long, device=self.device
            )
            loss = self.criterion(logits, targets, input_lengths, target_lengths)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Update weights
            self.optimizer.step()

            epoch_loss += loss.item()

            # Print batch progress
            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                print(
                    f"  Batch {batch_idx + 1}/{len(self.train_loader)} | Loss: {loss.item():.4f}"
                )

        # Calculate average epoch loss
        avg_loss = epoch_loss / len(self.train_loader)
        time_taken = time.time() - start_time

        return avg_loss, time_taken

    def validate(self) -> Tuple[float, List[str]]:
        """Run validation and compute metrics"""
        self.model.eval()
        epoch_loss = 0.0
        decoded_examples = []

        with torch.no_grad():
            for batch_idx, (images, targets, target_lengths) in enumerate(
                self.val_loader
            ):
                # Move to device
                images = images.to(self.device)
                targets = targets.to(self.device)
                target_lengths = target_lengths.to(self.device)

                # Forward pass
                features = self.model.cnn_encoder(images)
                logits = self.model.ctc_decoder(features)

                # Calculate loss
                input_lengths = torch.full(
                    (logits.size(0),),
                    logits.size(1),
                    dtype=torch.long,
                    device=self.device,
                )
                loss = self.criterion(logits, targets, input_lengths, target_lengths)

                epoch_loss += loss.item()

                # Decode predictions for a few samples
                if batch_idx == 0:
                    # Get first 8 samples in batch for visualization
                    sample_size = min(8, images.size(0))
                    sample_logits = logits[:sample_size]
                    decoded_texts = self.model.ctc_decoder.decode(sample_logits)
                    decoded_examples.extend(decoded_texts)

        # Calculate average validation loss
        avg_loss = epoch_loss / len(self.val_loader)

        return avg_loss, decoded_examples

    def train(self) -> Dict[str, List[float]]:
        """Main training loop"""
        print(f"Starting training for {self.max_epochs} epochs")

        for epoch in range(self.max_epochs):
            # Training phase
            train_loss, train_time = self.train_epoch()
            self.train_losses.append(train_loss)

            # Validation phase
            val_loss, decoded_examples = self.validate()
            self.val_losses.append(val_loss)

            # Get learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.learning_rates.append(current_lr)

            # Update learning rate
            if isinstance(self.scheduler, opt.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Print epoch results
            print(
                f"Epoch {epoch + 1}/{self.max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"LR: {current_lr:.6f} | "
                f"Time: {train_time:.2f}s"
            )

            # Print some decoded examples
            if decoded_examples:
                print("\nSample Predictions:")
                for i, text in enumerate(decoded_examples[:5]):
                    print(f"  Sample {i + 1}: {text}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0
                self._save_checkpoint(epoch, is_best=True)
                print(f"✅ New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.epochs_without_improvement += 1

            # Regular checkpoint saving (every 10 epochs)
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)

            # Early stopping
            if self.epochs_without_improvement >= self.early_stopping_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            print("-" * 80)

        # Save final model
        self._save_checkpoint(self.max_epochs - 1, is_final=True)

        # Plot training history
        self._plot_training_history()

        # Return training history
        return {
            "train_loss": self.train_losses,
            "val_loss": self.val_losses,
            "learning_rates": self.learning_rates,
        }

    def _save_checkpoint(
        self, epoch: int, is_best: bool = False, is_final: bool = False
    ) -> None:
        """Save model checkpoint"""
        if is_best:
            save_path = os.path.join(self.save_dir, "best_model.pth")
        elif is_final:
            save_path = os.path.join(self.save_dir, "final_model.pth")
        else:
            save_path = os.path.join(self.save_dir, f"checkpoint_epoch_{epoch + 1}.pth")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict()
                if hasattr(self.scheduler, "state_dict")
                else None,
                "train_loss": self.train_losses[-1]
                if self.train_losses
                else float("inf"),
                "val_loss": self.val_losses[-1] if self.val_losses else float("inf"),
            },
            save_path,
        )

    def _plot_training_history(self) -> None:
        """Plot training metrics"""
        plt.figure(figsize=(12, 10))

        # Plot training & validation loss
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, label="Training Loss")
        plt.plot(self.val_losses, label="Validation Loss")
        plt.title("Loss History")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # Plot learning rate
        plt.subplot(2, 1, 2)
        plt.plot(self.learning_rates)
        plt.title("Learning Rate Schedule")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.yscale("log")
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "training_history.png"))
        plt.close()


def train_license_plate_model(
    model,
    train_loader,
    val_loader,
    learning_rate=0.001,
    weight_decay=0.01,
    epochs=200,
    save_dir="checkpoints",
    early_stopping_patience=10,
    device=None,
):
    """Convenience function to set up and run the training process"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to device
    model = model.to(device)

    # Set up optimizer and scheduler
    optimizer, scheduler = adamw_setup(
        model, learning_rate=learning_rate, weight_decay=weight_decay, epochs=epochs
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        save_dir=save_dir,
        max_epochs=epochs,
        early_stopping_patience=early_stopping_patience,
    )

    # Start training
    history = trainer.train()

    return model, history
