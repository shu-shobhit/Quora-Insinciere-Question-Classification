import torch
train_config = {
    "dataset_path": "/kaggle/input/quora-insincere-questions-classification/train.csv",
    "model_card": "FacebookAI/roberta-base",
    "checkpoint_dir": "checkpoints",
    "pos_weight": [1, 20],
    "length_percentile": 99.9,
    "batch_size": 64,
    "epochs": 2,
    "seed": 42,
    "learning_rate": 1e-5,
    "lr_gamma": 0.9,
    "max_length": 62,
    "val_size": 0.2,
    "use_wandb": True,
    "wandb_project": "question-classification",
    "save_checkpoint_steps": 2000,
    "gradient_clip_norm": 1.0,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}

inference_config = {
    "model_path": "checkpoints/final_model.pth",
    "model_card": "FacebookAI/roberta-base",
    "max_length": 62,
    "batch_size": 128,
    "threshold": 0.5,  # Default threshold, can be overridden
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}