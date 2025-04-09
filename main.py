import os
import torch
import argparse
import pandas as pd
from model import BERT_MODEL
from dataset import prepare_dataloaders
from trainer import Trainer
from inference import Inferencer
from configurations import train_config, inference_config

def train_model(config):
    print(f"Starting training with {config['model_card']}...")
    
    train_dl, val_dl, _ = prepare_dataloaders(config)
    print(f"Train dataset size: {len(train_dl.dataset)}")
    print(f"Validation dataset size: {len(val_dl.dataset)}")
    
    model = BERT_MODEL(config["model_card"])
    print(f"Model initialized: {config['model_card']}")
    
    trainer = Trainer(model, train_dl, val_dl, config)
    history, model_path = trainer.train()
    
    print(f"Training completed. Model saved to {model_path}")
    return model_path

def run_inference(config, data_path=None):
    print(f"Starting inference with model from {config['model_path']}...")
    
    inferencer = Inferencer(config)
    
    if data_path is None:
        data_path = input("Enter path to test data CSV: ")
    
    results = inferencer.predict(data_path)
    
    save_option = input("Do you want to save predictions? (y/n): ")
    if save_option.lower() == 'y':
        output_path = input("Enter output file path (default: predictions.csv): ") or "predictions.csv"
        
        if isinstance(data_path, str) and os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            df = data_path
            
        df['prediction'] = results['predictions']
        df['probability'] = results['probabilities']
        
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
    
    return results

def main():
    print("Quora Insincere Questions Classification")
    print("----------------------------------------")
    
    parser = argparse.ArgumentParser(description='Run training or inference for question classification')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], 
                        help='Mode to run: train or inference')
    parser.add_argument('--data_path', type=str, help='Path to data file (for inference)')
    
    args = parser.parse_args()
    
    mode = args.mode
    if mode is None:
        mode = input("Select mode (train/inference): ").strip().lower()
        while mode not in ['train', 'inference']:
            print("Invalid mode. Please enter 'train' or 'inference'")
            mode = input("Select mode (train/inference): ").strip().lower()
    
    if mode == 'train':
        modify = input("Do you want to modify training config? (y/n): ").strip().lower()
        if modify == 'y':
            print("Current config:")
            for key, value in train_config.items():
                print(f"{key}: {value}")
            
            print("\nEnter new values (leave blank to keep current value):")
            for key in train_config:
                if key == 'device':
                    continue
                
                new_value = input(f"{key} [{train_config[key]}]: ")
                if new_value:
                    
                    if isinstance(train_config[key], bool):
                        train_config[key] = new_value.lower() == 'true'
                    elif isinstance(train_config[key], int):
                        train_config[key] = int(new_value)
                    elif isinstance(train_config[key], float):
                        train_config[key] = float(new_value)
                    elif isinstance(train_config[key], list):
                        
                        train_config[key] = eval(new_value)
                    else:
                        train_config[key] = new_value
        
        model_path = train_model(train_config)
        
        run_infer = input("Do you want to run inference with the trained model? (y/n): ").strip().lower()
        if run_infer == 'y':
            inference_config['model_path'] = model_path
            inference_config['model_card'] = train_config['model_card']
            
            data_path = args.data_path or input("Enter path to test data CSV: ")
            run_inference(inference_config, data_path)
    
    elif mode == 'inference':
        data_path = args.data_path
        run_inference(inference_config, data_path)
    
    print("Done!")

if __name__ == "__main__":
    main()