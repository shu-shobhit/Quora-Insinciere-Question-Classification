import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from rich.progress import Progress, TextColumn, BarColumn, SpinnerColumn
from model import BERT_MODEL
from dataset import QuestionsDataset

class Inferencer:
    def __init__(self, config):
        self.config = config
        self.device = config["device"]
        self.threshold = config.get("threshold", 0.5)
        
        self.model = BERT_MODEL(config["model_card"])
        self.model.load_state_dict(torch.load(config["model_path"], map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_card"], use_fast=True)
        
        print(f"Model loaded from {config['model_path']} to {self.device}")
        print(f"Using threshold: {self.threshold}")
    
    def prepare_dataloader(self, data_path):
        if isinstance(data_path, str):
            df = pd.read_csv(data_path)
        else:
            df = data_path
            
        is_test = 'target' not in df.columns
        
        dataset = QuestionsDataset(
            df, 
            self.tokenizer, 
            is_test=is_test, 
            max_length=self.config["max_length"]
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=False
        )
        
        return dataloader, is_test
    
    def predict(self, data_path):
        dataloader, is_test = self.prepare_dataloader(data_path)
        
        all_predictions = []
        all_probabilities = []
        all_targets = [] if not is_test else None
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        ) as progress:
            
            infer_task = progress.add_task(
                "[cyan]Running inference...", 
                total=len(dataloader)
            )
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    probabilities = torch.sigmoid(outputs.squeeze(1))
                    predictions = (probabilities > self.threshold).int()
                    
                    all_probabilities.extend(probabilities.cpu().numpy())
                    all_predictions.extend(predictions.cpu().numpy())
                    
                    if not is_test and 'targets' in batch:
                        all_targets.extend(batch["targets"].numpy())
                    
                    progress.update(infer_task, advance=1)
        
        results = {
            "predictions": np.array(all_predictions),
            "probabilities": np.array(all_probabilities),
        }
        
        if all_targets:
            results["targets"] = np.array(all_targets)
            from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
            
            results["metrics"] = {
                "accuracy": accuracy_score(results["targets"], results["predictions"]),
                "f1": f1_score(results["targets"], results["predictions"]),
                "precision": precision_score(results["targets"], results["predictions"], zero_division=0),
                "recall": recall_score(results["targets"], results["predictions"], zero_division=0)
            }
            
            print("\nPerformance Metrics:")
            for metric, value in results["metrics"].items():
                print(f"{metric.capitalize()}: {value:.4f}")
        
        return results