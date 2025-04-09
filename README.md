# Quora Insincere Question Classification

![Quora Logo](https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Quora_logo_2015.svg/250px-Quora_logo_2015.svg.png)

## Project Overview

This repository contains my solution for the Kaggle competition [Quora Insincere Questions Classification](https://www.kaggle.com/c/quora-insincere-questions-classification). The challenge focuses on identifying and filtering out toxic and misleading questions from the Quora platform.

### The Problem

Quora is a platform that empowers people to learn from each other through asking questions and connecting with others who provide unique insights. A significant challenge for the platform is to filter out "insincere" questions—those founded upon false premises or intended to make statements rather than seek genuine answers.

An insincere question may have these characteristics:

- Non-neutral or exaggerated tone targeting groups of people
- Rhetorical questions implying statements about groups
- Disparaging or inflammatory content
- Discriminatory ideas or stereotype confirmation
- Attacks against specific individuals or groups
- Outlandish premises about groups of people
- Disparagement of immutable characteristics
- Questions not grounded in reality or based on false information
- Sexual content used for shock value rather than seeking genuine answers

## Dataset

The dataset consists of over 1.3 million questions with binary labels indicating whether a question is sincere (0) or insincere (1).

### Data Description:

- **Training set**: 1,306,122 questions
- **Test set**: 56,370 questions
- **Features**:
  - `qid`: Unique question identifier
  - `question_text`: The text of the Quora question
  - `target`: Binary label (1 for insincere, 0 for sincere)

## Solution Approach

### Model Architecture

I implemented a deep learning approach using PyTorch with fine-tuned RoBERTa (FacebookAI/roberta-base) as the base model. The architecture includes:

- RoBERTa pre-trained language model for feature extraction
- A custom classification head with dropout for regularization
- Binary cross-entropy loss with logits for handling class imbalance

### Training Process

- Utilized stratified sampling to maintain class distribution
- Implemented gradient clipping to prevent exploding gradients
- Used AdamW optimizer with learning rate scheduling
- Applied exponential learning rate decay to optimize convergence

## Performance

- **Final F1 Score**: 0.70198 on the test set
- **Ranking**: 560 out of 1397 teams
- **Key Training Metrics**:
  - Best validation loss: 0.086472
  - Best F1 score: 0.72935 during validation
  - Convergence after 15999 steps of minibatch size 64

## Training Visualization

### Training Loss

![Training Loss](./Results/Train%20Loss.png)

### F1 Score on Val Set During Training

![F1 Score Evolution](./Results/F1%20Score%20on%20Val%20set.png)

### Threshold of obtaining Best F1 Score

![Threshold](./Results/Threshold%20of%20Obtaining%20best%20F1%20Score.png)

### Validation Loss

![Validation Loss](./Results/Validation%20Loss.png)

### Wandb Dashboard Link
[Dashboard_Link](https://wandb.ai/shobhitshukla6535-iit-kharagpur/question-classification/workspace?nw=nwusershobhitshukla6535)

## Implementation Details

The model implementation is divided into several modules:

- `configurations.py`: Configuration parameters for training and inference
- `dataset.py`: Data loading and preprocessing pipeline
- `model.py`: Model architecture definition
- `trainer.py`: Training loop with validation logic
- `inference.py`: Prediction functionality
- `main.py`: Entry point with CLI interface

### Key Technical Challenges

1. **Handling Class Imbalance**: Implemented weighted loss functions to address the imbalance between sincere and insincere questions.
2. **Exploding Gradients**: Identified and mitigated exploding gradients during training with gradient clipping (max norm of 5000).
3. **Optimization**: Dynamic learning rate scheduling to improve model convergence and performance.

## Running the Project

### Requirements

```
torch>=1.8.0
transformers>=4.5.0
pandas>=1.2.0
numpy>=1.19.0
scikit-learn>=0.24.0
rich>=10.0.0
wandb>=0.12.0 (optional)
```

### Training

```python
python main.py --mode train
```

### Inference

```python
python main.py --mode inference --data_path test.csv
```

## Future Improvements

- Experiment with ensemble methods combining multiple pre-trained language models

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Kaggle and Quora for providing the dataset and competition platform
- Hugging Face for the transformers library
