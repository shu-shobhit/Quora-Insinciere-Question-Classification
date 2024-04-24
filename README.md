# Quora-Insinciere-Question-Classification
Quora is a platform that empowers people to learn from each other. On Quora, people can ask questions and connect with others who contribute unique insights and quality answers. A key challenge is to weed out insincere questions -- those founded upon false premises, or that intend to make a statement rather than look for helpful answers.
<p> In this competition we have to predict whether a question asked on Quora is sincere or not.
</p>
An insincere question is defined as a question intended to make a statement rather than look for helpful answers. Some characteristics that can signify that a question is insincere:
  <li>Has a non-neutral tone
  <li>Has an exaggerated tone to underscore a point about a group of people
  <li> Is rhetorical and meant to imply a statement about a group of people
  <li> Is disparaging or inflammatory
  <li> Suggests a discriminatory idea against a protected class of people, or seeks confirmation of a stereotype
  <li>Makes disparaging attacks/insults against a specific person or group of people
  <li>Based on an outlandish premise about a group of people
  <li>Disparages against a characteristic that is not fixable and not measurable
  <li>Isn't grounded in reality
  <li>Based on false information, or contains absurd assumption
  <li>Uses sexual content (incest, bestiality, pedophilia) for shock value, and not to seek genuine answers
<p>
The training data includes the question that was asked, and whether it was identified as insincere (target = 1). The ground-truth labels contain some amount of noise: they are not guaranteed to be perfect. 
</p>

## File Description
train.csv - the training set <br>
test.csv - the test set <br>
sample_submission.csv - A sample submission in the correct format <br>

## Data fields
qid - unique question identifier <br>
question_text - Quora question text <br>
target - a question labeled "insincere" has a value of 1, otherwise 0 <br>

## Size of Dataset
train.shape
>(1306122, 3) <br>

test.shape
>(56370, 2)


## Model Description
Implemented a deep learning model using PyTorch and fine-tuned BERT architecture for the Quora Insincere Question Classification competition.

- Trained and validated the model on a dataset of 1.3 million questions, achieving an F1 score of 0.69.
- Identified and addressed the issue of exploding gradients during training using techniques such as gradient clipping with a max
norm of 5000, resulting in stable model training.
- Implemented a dynamic learning rate scheduling strategy to optimise model performance and got F1 score of 0.70198 on test
- Ranked 560 out of 1397 submissions in the Quora Insincere Question Classification competition.
