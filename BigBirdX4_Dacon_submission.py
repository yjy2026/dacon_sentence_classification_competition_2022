import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm.auto import tqdm
from transformers import *


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(7789) # some random seed


device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")


def replace_html(my_str):
    parseText = re.sub("&quot;", "\"", my_str)
    return parseText

def create_folds(data, num_splits):
    data["kfold"] = -1
    mskf = MultilabelStratifiedKFold(n_splits=num_splits, shuffle=True, random_state=8888)
    labels = ["문장", "유형", "극성", "시제", "확실성"]
    data_labels = data[labels].values
    for f, (t_, v_) in enumerate(mskf.split(data, data_labels)):
        data.loc[v_, "kfold"] = f
    return data

# naive train/test split
df = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

df = create_folds(df, 10)

for f in range(10):
    if f != 7:
        continue
    print(f"===== validating on fold {f+1} =====")
    train, val = df[df["kfold"] != f], df[df["kfold"] == f]

    # simple data preprocessing
    train_sentences = train["문장"].values
    train_Y1 = train["유형"].values
    train_Y2 = train["극성"].values
    train_Y3 = train["시제"].values
    train_Y4 = train["확실성"].values

    '''
    print("initial train dataset sizes")
    print(train_sentences.shape, train_Y1.shape, train_Y2.shape, train_Y3.shape, train_Y4.shape)
    add_train_sentences = []
    add_train_Y1 = []
    add_train_Y2 = []
    add_train_Y3 = []
    add_train_Y4 = []
    with open("backtranslated_sentences_dict_eng.pkl", "rb") as pkl_dict:
        eng_aug_dict = pickle.load(pkl_dict)
    for i in tqdm(range(len(train_sentences)), desc="augmenting data"):
        if train_sentences[i] in eng_aug_dict.keys():
                add_train_sentences.append(replace_html(eng_aug_dict[train_sentences[i]]))
                add_train_Y1.append(train_Y1[i])
                add_train_Y2.append(train_Y2[i])
                add_train_Y3.append(train_Y3[i])
                add_train_Y4.append(train_Y4[i])
    add_train_sentences = np.array(add_train_sentences)
    add_train_Y1 = np.array(add_train_Y1)
    add_train_Y2 = np.array(add_train_Y2)
    add_train_Y3 = np.array(add_train_Y3)
    add_train_Y4 = np.array(add_train_Y4)
    train_sentences = np.concatenate([train_sentences, add_train_sentences])
    train_Y1 = np.concatenate([train_Y1, add_train_Y1])
    train_Y2 = np.concatenate([train_Y2, add_train_Y2])
    train_Y3 = np.concatenate([train_Y3, add_train_Y3])
    train_Y4 = np.concatenate([train_Y4, add_train_Y4])

    print("dataset shape after augmentation!")
    print(train_sentences.shape, train_Y1.shape, train_Y2.shape, train_Y3.shape, train_Y4.shape)
    '''

    val_sentences = val["문장"].values
    val_Y1 = val["유형"].values
    val_Y2 = val["극성"].values
    val_Y3 = val["시제"].values
    val_Y4 = val["확실성"].values

    Y1_dict = {"대화형": 0, "사실형": 1, "예측형": 2, "추론형": 3}
    Y2_dict = {"긍정": 0, "미정": 1, "부정": 2}
    Y3_dict = {"과거": 0, "미래": 1, "현재": 2}
    Y4_dict = {"불확실": 0, "확실": 1}

    Y1_dict_rev = {0: "대화형", 1: "사실형", 2: "예측형", 3: "추론형"}
    Y2_dict_rev = {0: "긍정", 1: "미정", 2: "부정"}
    Y3_dict_rev = {0: "과거", 1: "미래", 2: "현재"}
    Y4_dict_rev = {0: "불확실", 1: "확실"}

    train_Y1_cat = []
    for i in range(len(train_Y1)):
        train_Y1_cat.append(Y1_dict[train_Y1[i]])

    train_Y2_cat = []
    for i in range(len(train_Y2)):
        train_Y2_cat.append(Y2_dict[train_Y2[i]])

    train_Y3_cat = []
    for i in range(len(train_Y3)):
        train_Y3_cat.append(Y3_dict[train_Y3[i]])

    train_Y4_cat = []
    for i in range(len(train_Y4)):
        train_Y4_cat.append(Y4_dict[train_Y4[i]])

    val_Y1_cat = []
    for i in range(len(val_Y1)):
        val_Y1_cat.append(Y1_dict[val_Y1[i]])

    val_Y2_cat = []
    for i in range(len(val_Y2)):
        val_Y2_cat.append(Y2_dict[val_Y2[i]])

    val_Y3_cat = []
    for i in range(len(val_Y3)):
        val_Y3_cat.append(Y3_dict[val_Y3[i]])

    val_Y4_cat = []
    for i in range(len(val_Y4)):
        val_Y4_cat.append(Y4_dict[val_Y4[i]])


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


    class WeightedLayerPooling(nn.Module):
        def __init__(self, num_hidden_layers, layer_start: int=4, layer_weights=None):
            super(WeightedLayerPooling, self).__init__()
            self.layer_start = layer_start
            self.num_hidden_layers = num_hidden_layers
            self.layer_weights = nn.Parameter(torch.tensor([1]*(num_hidden_layers+1 - layer_start), dtype=torch.float))
        def forward(self, all_hidden_states):
            all_layer_embedding = torch.stack(list(all_hidden_states), dim=0)
            all_layer_embedding = all_layer_embedding[self.layer_start:,:,:,:]
            weight_factor = self.layer_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(all_layer_embedding.size())
            weighted_average = (weight_factor * all_layer_embedding).sum(dim=0) / self.layer_weights.sum()
            return weighted_average


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

    class WeightedFocalLoss(nn.Module):
        def __init__(self, alpha, gamma=2):
            super(WeightedFocalLoss, self).__init__()
            self.alpha = alpha
            self.device = torch.device("cuda")
            self.alpha = self.alpha.to(self.device)
            self.gamma = gamma
        def forward(self, inputs, targets):
            CE_loss = nn.CrossEntropyLoss()(inputs, targets)
            targets = targets.type(torch.long)
            at = self.alpha.gather(0, targets.data.view(-1))
            pt = torch.exp(-CE_loss)
            F_loss = at * (1-pt)**self.gamma * CE_loss
            return F_loss.mean()

    def flat_accuracy(preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    train_input_ids, train_attn_masks = [], []
    for i in tqdm(range(len(train_sentences))):
        encoded_inputs = tokenizer(train_sentences[i], max_length=400, truncation=True, padding='max_length')
        train_input_ids.append(encoded_inputs["input_ids"])
        train_attn_masks.append(encoded_inputs["attention_mask"])

    val_input_ids, val_attn_masks = [], []
    for i in tqdm(range(len(val_sentences))):
        encoded_inputs = tokenizer(val_sentences[i], max_length=400, truncation=True, padding='max_length')
        val_input_ids.append(encoded_inputs["input_ids"])
        val_attn_masks.append(encoded_inputs["attention_mask"])

    train_input_ids = torch.tensor(train_input_ids, dtype=int)
    train_attn_masks = torch.tensor(train_attn_masks, dtype=int)

    val_input_ids = torch.tensor(val_input_ids, dtype=int)
    val_attn_masks = torch.tensor(val_attn_masks, dtype=int)

    train_Y1_labels = torch.tensor(train_Y1_cat, dtype=int)
    val_Y1_labels = torch.tensor(val_Y1_cat, dtype=int)

    train_Y2_labels = torch.tensor(train_Y2_cat, dtype=int)
    val_Y2_labels = torch.tensor(val_Y2_cat, dtype=int)

    train_Y3_labels = torch.tensor(train_Y3_cat, dtype=int)
    val_Y3_labels = torch.tensor(val_Y3_cat, dtype=int)

    train_Y4_labels = torch.tensor(train_Y4_cat, dtype=int)
    val_Y4_labels = torch.tensor(val_Y4_cat, dtype=int)

    batch_size = 16
    train_data = TensorDataset(train_input_ids, train_attn_masks, train_Y1_labels, train_Y2_labels, train_Y3_labels, train_Y4_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    val_data = TensorDataset(val_input_ids, val_attn_masks, val_Y1_labels, val_Y2_labels, val_Y3_labels, val_Y4_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    type_class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_Y1_cat), y=np.array(train_Y1_cat))
    type_class_weights = torch.tensor(type_class_weights, dtype=torch.float)
    type_loss_func = WeightedFocalLoss(alpha=type_class_weights)

    polarity_class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_Y2_cat), y=np.array(train_Y2_cat))
    polarity_class_weights = torch.tensor(polarity_class_weights, dtype=torch.float)
    polarity_loss_func = WeightedFocalLoss(alpha=polarity_class_weights)

    tense_class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_Y3_cat), y=np.array(train_Y3_cat))
    tense_class_weights = torch.tensor(tense_class_weights, dtype=torch.float)
    tense_loss_func = WeightedFocalLoss(alpha=tense_class_weights)

    certainty_class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(train_Y4_cat), y=np.array(train_Y4_cat))
    certainty_class_weights = torch.tensor(certainty_class_weights, dtype=torch.float)
    certainty_loss_func = WeightedFocalLoss(alpha=certainty_class_weights)

    criterion = {
        "type": nn.CrossEntropyLoss().to(device),
        "polarity": nn.CrossEntropyLoss().to(device),
        "tense": nn.CrossEntropyLoss().to(device),
        "certainty": nn.CrossEntropyLoss().to(device)
    }

    best_loss = 9999999999
    best_f1 = -99999999

    model = NeuralCLF()
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
    epochs = 5 #10
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)

    model.zero_grad()
    for epoch_i in tqdm(range(0,epochs), desc="Epochs", position=0, leave=True, total=epochs):
        train_loss = 0
        model.train()
        with tqdm(train_dataloader, unit = "batch") as tepoch:
            for step, batch in enumerate(tepoch):
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_type_labels, b_polarity_labels, b_tense_labels, b_certainty_labels = batch
                type_logit, polarity_logit, tense_logit, certainty_logit = model(b_input_ids, b_input_mask)
                loss = 0.25*criterion["type"](type_logit, b_type_labels) + 0.25*criterion["polarity"](polarity_logit, b_polarity_labels) + 0.25*criterion["tense"](tense_logit, b_tense_labels) + 0.25*criterion["certainty"](certainty_logit, b_certainty_labels)
                train_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                tepoch.set_postfix(loss=train_loss / (step+1))
                time.sleep(0.1)
        avg_train_loss = train_loss / len(train_dataloader)
        print(f"average train loss: {avg_train_loss}")
        val_loss = 0
        model.eval()
        type_preds, polarity_preds, tense_preds, certainty_preds = [], [], [], []
        type_labels, polarity_labels, tense_labels, certainty_labels = [], [], [], []
        for step, batch in tqdm(enumerate(val_dataloader), position=0, leave=True, total=len(val_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_type_labels, b_polarity_labels, b_tense_labels, b_certainty_labels = batch
            with torch.no_grad():
                type_logit, polarity_logit, tense_logit, certainty_logit = model(b_input_ids, b_input_mask)
                loss = 0.25*criterion["type"](type_logit, b_type_labels) + 0.25*criterion["polarity"](polarity_logit, b_polarity_labels) + 0.25*criterion["tense"](tense_logit, b_tense_labels) + 0.25*criterion["certainty"](certainty_logit, b_certainty_labels)
                val_loss += loss.item()

                type_preds += type_logit.argmax(1).detach().cpu().numpy().tolist()
                type_labels += b_type_labels.detach().cpu().numpy().tolist()

                polarity_preds += polarity_logit.argmax(1).detach().cpu().numpy().tolist()
                polarity_labels += b_polarity_labels.detach().cpu().numpy().tolist()

                tense_preds += tense_logit.argmax(1).detach().cpu().numpy().tolist()
                tense_labels += b_tense_labels.detach().cpu().numpy().tolist()

                certainty_preds += certainty_logit.argmax(1).detach().cpu().numpy().tolist()
                certainty_labels += b_certainty_labels.detach().cpu().numpy().tolist()

        type_f1 = f1_score(type_labels, type_preds, average="weighted")
        polarity_f1 = f1_score(polarity_labels, polarity_preds, average="weighted")
        tense_f1 = f1_score(tense_labels, tense_preds, average="weighted")
        certainty_f1 = f1_score(certainty_labels, certainty_preds, average="weighted")

        avg_val_loss = val_loss / len(val_dataloader)
        print(f"average val loss: {avg_val_loss}")
        print(f"유형 F1: {type_f1}")
        print(f"극성 F1: {polarity_f1}")
        print(f"시제 F1: {tense_f1}")
        print(f"확실성 F1: {certainty_f1}")
        f1_mult = type_f1 * polarity_f1 * tense_f1 * certainty_f1
        print(f"f1 mult : {f1_mult}")
        if f1_mult > best_f1:
            best_f1 = f1_mult
            torch.save(model.state_dict(), f"_BIGBIRDX4_Fold{f+1}_Epoch:{epoch_i+1}_loss:{avg_val_loss}_f1:{f1_mult}.pt")
