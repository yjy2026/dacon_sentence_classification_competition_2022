import numpy as np
import random
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm.auto import tqdm
from transformers import *
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler

import time
from datetime import datetime

from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# define tokenizer
tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")

# define model
class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    def forward(self, last_hidden_state, attention_masks):
        input_mask_expanded = attention_masks.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings

class MultiSampleDropout(nn.Module):
    def __init__(self, max_dropout_rate, num_samples, classifier):
        super(MultiSampleDropout, self).__init__()
        self.dropout = nn.Dropout
        self.classifier = classifier
        self.max_dropout_rate = max_dropout_rate
        self.num_samples = num_samples
    def forward(self, out):
        return torch.mean(torch.stack([self.classifier(self.dropout(p=self.max_dropout_rate)(out)) for _, rate in enumerate(np.linspace(0, self.max_dropout_rate, self.num_samples))], dim=0), dim=0)

class NeuralCLF(nn.Module):
    def __init__(self, plm="monologg/kobigbird-bert-base", type_classes=4, polarity_classes=3, tense_classes=3, certainty_classes=2):
        super(NeuralCLF, self).__init__()
        self.type_classes = type_classes
        self.polarity_classes = polarity_classes
        self.tense_classes = tense_classes
        self.certainty_classes = certainty_classes
        self.config = AutoConfig.from_pretrained(plm)

        self.type_lm = AutoModel.from_pretrained(plm, config=self.config)
        self.polarity_lm = AutoModel.from_pretrained(plm, config=self.config)
        self.tense_lm = AutoModel.from_pretrained(plm, config=self.config)
        self.certainty_lm = AutoModel.from_pretrained(plm, config=self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(plm)
        self.mean_pooler = MeanPooling()

        self.type_fc = nn.Linear(self.config.hidden_size, self.type_classes)
        self._init_weights(self.type_fc)
        self.type_multi_dropout = MultiSampleDropout(0.2, 8, self.type_fc)

        self.polarity_fc = nn.Linear(self.config.hidden_size, self.polarity_classes)
        self._init_weights(self.polarity_fc)
        self.polarity_multi_dropout = MultiSampleDropout(0.2, 8, self.polarity_fc)

        self.tense_fc = nn.Linear(self.config.hidden_size, self.tense_classes)
        self._init_weights(self.tense_fc)
        self.tense_multi_dropout = MultiSampleDropout(0.2, 8, self.tense_fc)

        self.certainty_fc = nn.Linear(self.config.hidden_size, self.certainty_classes)
        self._init_weights(self.certainty_fc)
        self.certainty_multi_dropout = MultiSampleDropout(0.2, 8, self.certainty_fc)


    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, input_ids, attn_masks):
        type_x = self.type_lm(input_ids, attn_masks)[0]
        type_x = self.mean_pooler(type_x, attn_masks)

        polarity_x = self.polarity_lm(input_ids, attn_masks)[0]
        polarity_x = self.mean_pooler(polarity_x, attn_masks)

        tense_x = self.tense_lm(input_ids, attn_masks)[0]
        tense_x = self.mean_pooler(tense_x, attn_masks)

        certainty_x = self.certainty_lm(input_ids, attn_masks)[0]
        certainty_x = self.mean_pooler(certainty_x, attn_masks)

        out_x = type_x + polarity_x  + tense_x + certainty_x

        type_output = self.type_multi_dropout(out_x) + self.type_multi_dropout(type_x)
        polarity_output = self.polarity_multi_dropout(out_x) + self.polarity_multi_dropout(polarity_x)
        tense_output = self.tense_multi_dropout(out_x) + self.tense_multi_dropout(tense_x)
        certainty_output = self.certainty_multi_dropout(out_x) + self.certainty_multi_dropout(certainty_x)
        return type_output, polarity_output, tense_output, certainty_output


# naive train/test split
df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# define dictionary
Y1_dict = {"대화형": 0, "사실형": 1, "예측형": 2, "추론형": 3}
Y2_dict = {"긍정": 0, "미정": 1, "부정": 2}
Y3_dict = {"과거": 0, "미래": 1, "현재": 2}
Y4_dict = {"불확실": 0, "확실": 1}

Y1_dict_rev = {0: "대화형", 1: "사실형", 2: "예측형", 3: "추론형"}
Y2_dict_rev = {0: "긍정", 1: "미정", 2: "부정"}
Y3_dict_rev = {0: "과거", 1: "미래", 2: "현재"}
Y4_dict_rev = {0: "불확실", 1: "확실"}

test_sentences = test["문장"].values

test_input_ids, test_attn_masks = [], []
for i in tqdm(range(len(test_sentences))):
    encoded_inputs = tokenizer(test_sentences[i], max_length=400, truncation=True, padding='max_length')
    test_input_ids.append(encoded_inputs["input_ids"])
    test_attn_masks.append(encoded_inputs["attention_mask"])

test_input_ids = torch.tensor(test_input_ids, dtype=int)
test_attn_masks = torch.tensor(test_attn_masks, dtype=int)

batch_size = 16
test_data = TensorDataset(test_input_ids, test_attn_masks)
test_sampler = SequentialSampler(test_data)
test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

chkpts = ["BIGBIRDX4_Fold1_Epoch:4_loss:0.24398749192746785_f1:0.7416115893330479.pt",
          "BIGBIRDX4_Fold2_Epoch:3_loss:0.23193542182875368_f1:0.7202878180449389.pt",
          "BIGBIRDX4_Fold3_Epoch:3_loss:0.22103419025930074_f1:0.7359621893515262.pt",
          "BIGBIRDX4_Fold4_Epoch:3_loss:0.2187729666296106_f1:0.7432139671834832.pt",
          "BIGBIRDX4_Fold5_Epoch:3_loss:0.20905456045427576_f1:0.7515942017970072.pt",
          "BIGBIRDX4_Fold6_Epoch:3_loss:0.20471724212312928_f1:0.7409858388659708.pt",
          "BIGBIRDX4_Fold7_Epoch:3_loss:0.2157047412608965_f1:0.7387295744716654.pt",
          "_BIGBIRDX4_Fold8_Epoch:3_loss:0.21458650118886277_f1:0.7386543217890885.pt",
          "_BIGBIRDX4_Fold9_Epoch:3_loss:0.20604668163622802_f1:0.7409611653177859.pt",
          "_BIGBIRDX4_Fold10_Epoch:3_loss:0.21317914083528405_f1:0.7242800728382435.pt"]

all_predictions = [] # contains 10 arrays

for idx, chkpt in enumerate(chkpts):
        model = NeuralCLF()
        checkpoint = torch.load(chkpt)
        model.load_state_dict(checkpoint)
        model.eval()
        model.cuda()

        predictions = []
        type_preds, polarity_preds, tense_preds, certainty_preds = [], [], [], []
        for step, batch in tqdm(enumerate(test_dataloader), position=0, leave=True, total=len(test_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask = batch
                with torch.no_grad():
                        type_logits, polarity_logits, tense_logits, certainty_logits = model(b_input_ids, b_input_mask)
                type_logits = nn.Softmax()(type_logits).detach().cpu().numpy()
                polarity_logits = nn.Softmax()(polarity_logits).detach().cpu().numpy()
                tense_logits = nn.Softmax()(tense_logits).detach().cpu().numpy()
                certainty_logits = nn.Softmax()(certainty_logits).detach().cpu().numpy()

                for type_logit, polarity_logit, tense_logit, certainty_logit in zip(type_logits, polarity_logits, tense_logits, certainty_logits):
                        predictions.append([type_logit, polarity_logit, tense_logit, certainty_logit])
        all_predictions.append(predictions)

print("logits from all models calculated!")

type_list, polarity_list, tense_list, certainty_list = [], [], [], []
for i in range(len(all_predictions[0])):
        types = np.zeros(4)
        for k in range(10):
                types += all_predictions[k][i][0]
        types /= 10
        type_list.append(Y1_dict_rev[np.argmax(types)])

        polarities = np.zeros(3)
        for k in range(10):
                polarities += all_predictions[k][i][1]
        polarities /= 10
        polarity_list.append(Y2_dict_rev[np.argmax(polarities)])

        tenses = np.zeros(3)
        for k in range(10):
                tenses += all_predictions[k][i][2]
        tenses /= 10
        tense_list.append(Y3_dict_rev[np.argmax(tenses)])

        certainties = np.zeros(2)
        for k in range(10):
                certainties += all_predictions[k][i][3]
        certainties /= 10
        certainty_list.append(Y4_dict_rev[np.argmax(certainties)])


answers = []

for str_type, str_polarity, str_tense, str_certainty in zip(type_list, polarity_list, tense_list, certainty_list):
        answers.append(str_type + "-" + str_polarity + "-" + str_tense + "-" + str_certainty)

submission = pd.read_csv("sample_submission.csv")
submission["label"] = answers

print("saving prediction results...")

submission.to_csv("BigBirdX4_ensemble.csv",index=False)

print("done!")
