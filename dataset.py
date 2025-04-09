import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

class QuestionsDataset(Dataset):
    def __init__(self, dataset, tokenizer, is_test=False, max_length=62):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        text = str(self.dataset.iloc[idx, 1])

        if not self.is_test:
            target = self.dataset.iloc[idx, 2]
        
        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        if self.is_test:
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
            }
        else:
            return {
                "input_ids": encoding["input_ids"].squeeze(),
                "attention_mask": encoding["attention_mask"].squeeze(),
                "targets": torch.FloatTensor([target]),
            }

def get_max_length(df_train, tokenizer, length_percentile=99.9):
    """Calculate an appropriate max length based on data distribution"""
    df_train["question_length"] = tokenizer(
        df_train.question_text.tolist(), truncation=True
    )["input_ids"]
    df_train["question_length"] = df_train["question_length"].apply(lambda x: len(x))
    max_length = np.percentile(df_train["question_length"], length_percentile)
    return int(max_length)

def prepare_dataloaders(config):
    """Prepare train and validation dataloaders based on configuration"""
    df_train = pd.read_csv(config["dataset_path"])
    df_train.target = df_train.target.astype("int16")
    
    tokenizer = AutoTokenizer.from_pretrained(config["model_card"], use_fast=True)
    
    max_length = config.get("max_length") 
    if not max_length and config.get("length_percentile"):
        max_length = get_max_length(df_train, tokenizer, config["length_percentile"])
    
    train_df, val_df = train_test_split(
        df_train,
        stratify=df_train.target,
        test_size=config["val_size"],
        random_state=config["seed"],
    )

    train_ds = QuestionsDataset(
        train_df, tokenizer, is_test=False, max_length=max_length
    )
    val_ds = QuestionsDataset(
        val_df, tokenizer, is_test=False, max_length=max_length
    )

    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=config["batch_size"])
    
    return train_dl, val_dl, tokenizer