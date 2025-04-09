import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ExponentialLR
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn, MofNCompleteColumn

try:
    import wandb
except ImportError:
    print("Warning: wandb not installed. WandB logging will be disabled.")
    wandb = None

class Trainer:
    def __init__(self, model, train_dl, val_dl, config):
        self.model = model
        self.train_dl = train_dl
        self.val_dl = val_dl
        self.config = config
        self.device = config["device"]
        self.use_wandb = config.get("use_wandb", False)
        
        self.model.to(self.device)
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config["learning_rate"])
        self.scheduler = ExponentialLR(self.optimizer, gamma=config["lr_gamma"])
        
        self.checkpoint_dir = config["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.best_val_loss = float('inf')
        self.best_model_path = None
        
        if self.use_wandb and wandb is not None:
            wandb.init(
                project=config["wandb_project"],
                config={
                    "epochs": config["epochs"],
                    "batch_size": config["batch_size"],
                    "learning_rate": config["learning_rate"],
                    "model": config["model_card"],
                },
            )

    @staticmethod
    def find_best_f1(outputs, labels):
        tmp = [0, 0, 0] 
        threshold = 0

        for tmp[0] in np.arange(0.25, 0.85, 0.01):
            tmp[1] = f1_score(labels, outputs > tmp[0], zero_division=0)
            if tmp[1] > tmp[2]:
                threshold = tmp[0]
                tmp[2] = tmp[1]

        return tmp[2], threshold

    def train(self):
        torch.manual_seed(self.config["seed"])
        history = {"Train_Loss": [], "Val_Loss": [], "F1": []}
        agg_loss = 0
        step = 0

        progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            SpinnerColumn(),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        )

        with progress:
            epoch_progress = progress.add_task(
                "[cyan] Epochs...: ", total=self.config["epochs"], start=True
            )

            for epoch in range(self.config["epochs"]):
                progress.update(epoch_progress, advance=1)
                batch_progress = progress.add_task(
                    "[blue] Training :", total=len(self.train_dl)
                )

                self.model.train()
                for i, batch in enumerate(self.train_dl):
                    input_ids, attention_mask, targets = (
                        batch["input_ids"].to(self.device),
                        batch["attention_mask"].to(self.device),
                        batch["targets"].to(self.device).squeeze(1),
                    )

                    self.optimizer.zero_grad()
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits.squeeze(1), targets)
                    agg_loss += loss.item()
                    loss.backward()

                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        max_norm=self.config["gradient_clip_norm"]
                    )

                    self.optimizer.step()

                    avg_loss = agg_loss / (step + 1)
                    history["Train_Loss"].append(avg_loss)
                    
                    if self.use_wandb and wandb is not None:
                        wandb.log({"Train Loss": avg_loss}, step=step + 1)

                    progress.update(
                        batch_progress,
                        advance=1,
                        description=f"Train loss: {avg_loss:.4f}",
                    )

                    step = step + 1

                    if (step + 1) % self.config["save_checkpoint_steps"] == 0:
                        checkpoint_path = os.path.join(
                            self.checkpoint_dir, f"model_epoch_{epoch + 1}_step_{step + 1}.pth"
                        )
                        self.scheduler.step()

                        if self.use_wandb and wandb is not None:
                            wandb.log({"last_lr": self.scheduler.get_last_lr()[0]}, step=step)

                        torch.save(
                            {
                                "epoch": epoch + 1,
                                "step": step + 1,
                                "model_state_dict": self.model.state_dict(),
                                "optimizer_state_dict": self.optimizer.state_dict(),
                                "loss": history["Train_Loss"][-1],
                            },
                            checkpoint_path,
                        )
                        progress.console.print(f"Checkpoint saved at {checkpoint_path}")
                        
                        val_loss, best_f1, threshold = self.validate(progress)
                        progress.console.print(
                            f"Learning Rate after {step} steps: {self.scheduler.get_last_lr()[0]} \n",
                            f"Train loss: {history['Train_Loss'][-1]:.4f} \n",
                            f"Validation Loss: {val_loss}, Best F1 Score: {best_f1}, Threshold: {threshold}\n",
                        )
                        
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.best_model_path = os.path.join(
                                self.checkpoint_dir, f"best_model_epoch_{epoch + 1}_step_{step + 1}.pth"
                            )
                            
                            torch.save(
                                {
                                    "epoch": epoch + 1,
                                    "step": step + 1,
                                    "model_state_dict": self.model.state_dict(),
                                    "optimizer_state_dict": self.optimizer.state_dict(),
                                    "val_loss": val_loss,
                                    "f1_score": best_f1,
                                    "threshold": threshold,
                                },
                                self.best_model_path,
                            )
                            
                            progress.console.print(f"[bold green]New best model saved with val_loss: {val_loss:.4f} at {self.best_model_path}")
                            
                            if self.use_wandb and wandb is not None:
                                wandb.log(
                                    {
                                        "Best Validation Loss": val_loss,
                                        "Best F1 Score": best_f1,
                                        "Best Threshold": threshold,
                                    },
                                    step=step,
                                )
                                wandb.save(self.best_model_path)
                        
                        if self.use_wandb and wandb is not None:
                            wandb.log(
                                {
                                    "Validation Loss": val_loss,
                                    "F1 Score": best_f1,
                                    "at_threshold": threshold,
                                },
                                step=step,
                            )
                        
                        history["Val_Loss"].append(val_loss)
                        history["F1"].append(best_f1)

                progress.remove_task(batch_progress)

        history_path = os.path.join(self.checkpoint_dir, "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f)

        if self.use_wandb and wandb is not None:
            wandb.save(history_path)

        model_path = os.path.join(self.checkpoint_dir, "final_model.pth")
        torch.save(self.model.state_dict(), model_path)

        if self.use_wandb and wandb is not None:
            wandb.save(model_path)
            print("Training complete! History and model saved to W&B.")
            wandb.finish()
        else:
            print("Training complete! History and model saved locally.")
            
        return history, {
            "final_model_path": model_path,
            "best_model_path": self.best_model_path,
            "best_val_loss": self.best_val_loss
        }

    def validate(self, progress):
        self.model.eval()
        agg_val_loss = 0
        all_logits, all_targets = [], []
        val_progress = progress.add_task("[Red] Validating :", total=len(self.val_dl))
        
        with torch.no_grad():
            for i, batch in enumerate(self.val_dl):
                input_ids, attention_mask, targets = (
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                    batch["targets"].to(self.device).squeeze(1),
                )

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits.squeeze(1), targets)
                agg_val_loss += loss.item()
                progress.update(
                    val_progress,
                    advance=1,
                    description=f"Validated {i + 1} Batches",
                )
                all_logits.append(logits.squeeze(1).cpu())
                all_targets.append(targets.cpu())

        all_logits = torch.cat(all_logits, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        val_loss = agg_val_loss / len(self.val_dl)
        best_f1, threshold = self.find_best_f1(
            torch.sigmoid(all_logits).numpy(), all_targets.numpy()
        )

        progress.remove_task(val_progress)
        return val_loss, best_f1, threshold