import os

OUTPUT_DIR = './output/'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ====================================================
# CFG
# ====================================================
class CFG:
    #! Edit Model Paths Accordingly
    model_dir='Jigsaw\deberta_base'
    num_workers=4
    model="deberta-v3-base"
    batch_size=128
    fc_dropout=0.00001
    text="text"
    target="target"
    target_size=1
    head=32
    tail=32
    seed=2021
    n_fold=8


CFG.max_len = CFG.head + CFG.tail

import os
import gc
import re
import sys
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.ERROR)
import scipy as sp
import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig 
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_score(df):
    score = len(df[df['less_toxic_pred'] < df['more_toxic_pred']]) / len(df)
    return score

def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=2021)

tokenizer = AutoTokenizer.from_pretrained(CFG.model, lowercase=True)
tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
CFG.tokenizer = tokenizer

df = pd.read_csv('Jigsaw\Jigsaw_Rate_Severity_of_Toxic_Comments\comments_to_score.csv')
sub  = pd.read_csv('Jigsaw\Jigsaw_Rate_Severity_of_Toxic_Comments\sample_submission.csv')

def prepare_input(text, cfg):
    if cfg.tail == 0:
        inputs = cfg.tokenizer.encode_plus(text, 
                                           return_tensors=None, 
                                           add_special_tokens=True, 
                                           max_length=cfg.max_len,
                                           pad_to_max_length=True,
                                           truncation=True)
        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)
    else:
        inputs = cfg.tokenizer.encode_plus(text,
                                           return_tensors=None, 
                                           add_special_tokens=True, 
                                           truncation=True)
        for k, v in inputs.items():
            v_length = len(v)
            if v_length > cfg.max_len:
                v = np.hstack([v[:cfg.head], v[-cfg.tail:]])
            if k == 'input_ids':
                new_v = np.ones(cfg.max_len) * cfg.tokenizer.pad_token_id
            else:
                new_v = np.zeros(cfg.max_len)
            new_v[:v_length] = v 
            inputs[k] = torch.tensor(new_v, dtype=torch.long)
    return inputs

class TestDataset(Dataset):
    def __init__(self, cfg, df):
        self.cfg = cfg
        self.text = df[cfg.text].fillna("none").values

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        inputs = prepare_input(text, self.cfg)
        return inputs

class AttentionHead(nn.Module):
    def __init__(self, in_features, hidden_dim):
        super().__init__()
        self.in_features = in_features
        self.middle_features = hidden_dim
        self.W = nn.Linear(in_features, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        self.out_features = hidden_dim

    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector

class CustomModel(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        self.head = AttentionHead(self.config.hidden_size,self.config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.config.hidden_size,1)

    def forward(self, xb):
        x = self.model(**xb)[0]
        x = self.head(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

class CustomModel_Legacy(nn.Module):
    def __init__(self, cfg, config_path=None, pretrained=False):
        super().__init__()
        self.cfg = cfg
        self.config = AutoConfig.from_pretrained(cfg.model, output_hidden_states=True)
        self.model = AutoModel.from_pretrained(cfg.model, config=self.config)
        self.fc_dropout = nn.Dropout(cfg.fc_dropout)
        self.fc = nn.Linear(self.config.hidden_size, cfg.target_size)
        
    def feature(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs[0]
        feature = torch.mean(last_hidden_states, 1)
        return feature

    def forward(self, inputs):
        feature = self.feature(inputs)
        output = self.fc(self.fc_dropout(feature))
        return output

def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to('cpu').numpy())
    predictions = np.concatenate(preds)
    return predictions

test_dataset = TestDataset(CFG, df)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, 
                         num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
config_path = CFG.model_dir+"config.pth"
predictions = []
for fold in range(CFG.n_fold):
    model = CustomModel(CFG, config_path=config_path, pretrained=False)
    state = torch.load(CFG.model_dir+f"/jigsaw_fold{fold}_best.pth", map_location=torch.device('cpu'))
    model.load_state_dict(state['model'])
    prediction = inference_fn(test_loader, model, device)
    predictions.append(prediction)
    del model, state; gc.collect()
    torch.cuda.empty_cache()

preds_deberta = np.mean(predictions, axis=0).flatten()