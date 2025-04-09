config = {
    "model_name": "FacebookAI/roberta-base",
    "batch_size": 64,
    "learning_rate": 1.5e-5,
    "num_epochs":2,
    "max_length":64,
    "checkpoint_dir": "./checkpoints",
    "wandb_enabled": True,
    "wandb_project_name": "Quora_Insincere_Questions_Classification",
    "log_interval": 2000,
    
}