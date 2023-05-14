# Jigsaw Rate Severity of Toxic Comments

## Overview

<hr/>

Having already hosted 3 previous competitions, Jigsaw's 4th competition is like none before, both in terms of data provided and final private leaderboard shakeup (with most of the top 50 finishers moving up 1000+ positions from public leaderboard scores).

A brief overview of the competition can be found [here](https://www.kaggle.com/c/jigsaw-toxic-severity-rating/overview).

Very much unlike most of the NLP competitions, this competition does not provide training data and only provides a validation set, hoping to encourage participants to share and use open source data such as previous iterations of Jigsaw competitions as train sets. This paints a more open ended problem as compared to most other competitions. Funny enough, I was thrown a very similar style of open ended problem at work quite a few months back with no training data and no examples of test set data.

This open ended nature to the problems allow for more space and creativity to explore but in turn, made it very difficult to track your experimental scores and progress due to a larger unknown factor. This arises as no one can be certain of the data distribution of the private leaderboards and with such large disparities in the Public Leaderboard scores and Cross Validation (CV) scores, it was difficult (at least for me) to judge what was working and what was not.


## Data and Cross Validation (CV)

<hr/>

### **Data**

The Data I chose to use for the competition can be found below :

   1. [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/julian3833/jigsaw-toxic-comment-classification-challenge)

   2. [Jigsaw Multilingual Toxic Comment Classification](https://www.kaggle.com/julian3833/jigsaw-multilingual-toxic-comment-classification?select=jigsaw-toxic-comment-train.csv)
   
   3. [Ruddit Jigsaw Dataset](https://www.kaggle.com/rajkumarl/ruddit-jigsaw-dataset)
   
   4. [Malignant Comment Classification](https://www.kaggle.com/surekharamireddy/malignant-comment-classification)

### **Cross Validation (CV)**

Since the LB and CV scores are essentially inversely related (from my experiments), I decided on measuring performance of my experiment with a mix of CV and LB score `0.25 LB + 0.75 CV`. UnionFind and StratifiedKFolds were used for training

## Model  

<hr/>

### **Transformers**

I started the competition early but took a long break and came back 2 weeks before the competition ended so I did not have much time to explore model variety. (And I only trained models on Kaggle GPU) Hence I only trained the common and known to reliable transformer varients. 

Throughout the competition I explored BCE and Margin Ranking Loss, but in my case MRL always performed better than BCE.

The Transformer Model Performances can be found below :

| No. | Model | Dataset | CV | Loss Fn |
|---|---|---|---|---|
| 1 | Roberta Base | Toxic + Multilingual + Ruddit + Malignant | 0.69569 | Margin Ranking Loss |
| 2 | Roberta Base | Toxic + Multilingual + Ruddit + Malignant | 0.69465 | BCE |
| 3 | Roberta Large | Toxic + Multilingual + Ruddit + Malignant | 0.70021 | Margin Ranking Loss |
| 4 | Roberta Large | Toxic + Multilingual + Ruddit + Malignant | 0.69615 | BCE |
| 5 | Deberta Base | Toxic + Multilingual + Ruddit + Malignant | 0.70130 | Margin Ranking Loss |
| 6 | Deberta Base | Toxic + Multilingual + Ruddit + Malignant | 0.70101 | BCE |
| 7 | Deberta Base | Toxic + Multilingual + Malignant | 0.70115 | Margin Ranking Loss |
| 8 | Deberta Base | Toxic + Multilingual + Malignant | 0.69867 | BCE |

### **Linear Models**

Contrary to the general direction of discussions, I felt like linear models were important (to an extent). Linear Models performed well on LB but badly on CV while Transformers performed badly on LB but well on CV. Although it makes sense to trust your CV, I believe that it may not completely reliable in this case.

Although LB is only 5% of the final Private LB and we cant judge for the rest of the 95% of data, it is not useless as well, it gives an idea of generally what type of data / data distribution is like for our final private leaderboard. This is especially important as for this competition your distribution of train set is likely to be very different from the distribution of the test sets you will be evaluated on, (since you have to choose the training data yourself) and you are unsure if the validation data given has a similar distribution to the leaderboard data, hence on the final day I convinced myself to ensemble 1 Linear TFIDF model to my RoBERTa ensemble and selected it as my 2nd submission even though it did not have the best CV / LB score (The Linear Model gave me a small boost in score and if not for the last minute discussion I would have slipped to 60+ position).


| No. | Model | Dataset | CV |
|---|---|---|---|
| 9 | TF-IDF Ridge Regression | Toxic + Multilingual + Ruddit + Malignant | 0.67844
| 10 | TF-IDF Ridge Regression | Toxic + Multilingual + Malignant | 0.65105

* Linear Model is Heavily Inspired by @readoc's implementation [here](https://www.kaggle.com/readoc/toxic-linear-model-pseudo-labelling-lb-0-864) (I personally only did some hyperparameter tuning and data engineering changes)

***Final Submission Consists of Model No : 1, 3, 5, 9 (Equal Weight)***

### **Fine Tuning**

I spent most of my time tuning the Learning Rates and Decay Rates of my RoBERTa models, on hindsight, I could have spent more time om dataset preparation and playing around with Custom Loss Functions instead (wanted to do but did not have much time). Ultimately I think it was not very effective in increasing my CV but nevertheless did boost it by a small amount.


## What didnt work 

- MLM
- Boosted Trees (Funny enough it was the first thing I tried but couldnt make it work. Other competitors have found success with this and can be found [here](https://www.kaggle.com/c/jigsaw-toxic-severity-rating/discussion/306074).

**Only the Transformer Inference Code and Model Checkpoints Datasets will be shared in this repository. The inspiration behind the Linear Model can be found [here](https://www.kaggle.com/readoc/toxic-linear-model-pseudo-labelling-lb-0-864)**

**Model Checkpoint Locations :**

1. [Roberta Base](https://www.kaggle.com/toxicmaze/jigsaw-toxicrudditmultilingual-roberta-ckpt)

2. [Roberta Large](https://www.kaggle.com/toxicmaze/robertal-lr-1e5-1e6)

3. [Deberta Base](https://www.kaggle.com/toxicmaze/jigsaw-deberta-base)






