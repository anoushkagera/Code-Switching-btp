# -*- coding: utf-8 -*-
"""Code_Mixed_SA

Automatically generated by Colaboratory.
"""

# Copyright (c) Microsoft Corporation. Licensed under the MIT license.

DATA_PATH = '/home/bhavya24/Translated_Data/Sentiment_EN_HI/'
model_checkpoint = "cardiffnlp/bertweet-base-sentiment"
BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
MAX_SEQ_LEN = 128

# Commented out IPython magic to ensure Python compatibility.
# %%capture
# !pip install transformers

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Finetuning for SA')
    parser.add_argument('-n', '--experiment_id', type=str, required=True)
    return parser.parse_args()

args = parse_args()
EXPERIMENT_ID = args.experiment_id
TESTING=False

import torch
import os
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from tqdm.notebook import tqdm
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from torch.nn import CrossEntropyLoss

os.environ["HF_HOME"]="/scratch/huggingface_cache/"
os.makedirs(f'/scratch/bhavya24/{EXPERIMENT_ID}')

def get_labels(data_dir):
    all_path = os.path.join(data_dir, "all.txt")
    labels = []
    with open(all_path, "r") as infile:
        lines = infile.read().strip().split('\n')

    for line in lines:
        splits = line.split('\t')
        label = splits[-1]
        if label not in labels:
            labels.append(label)
    return labels

label_list = sorted(get_labels(DATA_PATH))
print(label_list)

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, normalization=True)#, use_fast=True)

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint)
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    model = torch.nn.DataParallel(model)
model = model.to(device)

def read_examples_from_file(data_dir, mode):
    file_path = os.path.join(data_dir, "{}.txt".format(mode))
    examples = []
    label_map = {label: i for i, label in enumerate(label_list)}
    with open(file_path, 'r') as infile:
        lines = infile.read().strip().split('\n')
    for line in lines:
        x = line.split('\t')
        assert len(x) == 2
        text = x[0]
        label = label_map[x[1]]
        examples.append({'text': text, 'label': label})
    return examples

class CustomDataset(Dataset):
    def __init__(self, tokenizer, examples):
        self.tokenizer = tokenizer
        self.data = examples
        self.max_len = MAX_SEQ_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        inputs = self.tokenizer.encode_plus(
            self.data[index]['text'],
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            # return_token_type_ids=True,
            truncation=True,
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        # token_type_ids = inputs["token_type_ids"]
        return {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            # 'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(self.data[index]['label'], dtype=torch.long)
        }

train_dataset = CustomDataset(tokenizer, read_examples_from_file(DATA_PATH, "train"))
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE)

valid_dataset = CustomDataset(tokenizer, read_examples_from_file(DATA_PATH, "validation"))
valid_dataloader = DataLoader(valid_dataset, shuffle=False, batch_size=EVAL_BATCH_SIZE)

test_dataset = CustomDataset(tokenizer, read_examples_from_file(DATA_PATH, "test"))
test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=EVAL_BATCH_SIZE)

def save_model(model, name, data):
    DIR = f'/scratch/bhavya24/{EXPERIMENT_ID}/'
    # model.save_pretrained(f'{DIR}/model_{name}')
    torch.save(model.module.state_dict(), f'{DIR}/model_{name}.pt')
    with open(f'{DIR}/valid_{name}.txt', 'w') as f:
        pred_labels = [label_list[x] for x in data['valid_preds']]
        f.write('\n'.join(pred_labels))
    with open(f'{DIR}/test_{name}.txt', 'w') as f:
        pred_labels = [label_list[x] for x in data['test_preds']]
        f.write('\n'.join(pred_labels))
    print('Model stored!', flush=True)

class ScorePreds:
    def __init__(self, keep_model=False):
        self.best_metrics = [0]
        self.best_model_cnt = 0
        self.test_score = 0

    def update_best_score(self, model, metrics, preds):
        if metrics[0] <= self.best_metrics[0]:
            return
        print(f'Exceeded the past model with {self.best_metrics[0]} with a score of {metrics[0]}', flush=True)
        print(metrics, flush=True)
        test_preds, test_gt = validate(test_dataloader, fast=False)
        self.test_score = f1_score(test_gt, test_preds, average='weighted', labels=range(len(label_list)))
        print(self.test_score)
        data = {
            'valid_preds': preds,
            'test_preds': test_preds,
        }
        save_model(model, f'nli_{self.best_model_cnt}', data)
        self.best_model_cnt += 1
        self.best_metrics = metrics

    def calc_score(self, model, preds, gt):
        assert len(preds) == len(gt)
        metrics = (
            f1_score(gt, preds, average='weighted', labels=range(len(label_list))),
            f1_score(gt, preds, average='micro', labels=range(len(label_list))),
            f1_score(gt, preds, average='macro', labels=range(len(label_list))),
            accuracy_score(gt, preds) * 100,
            confusion_matrix(gt, preds),
        )

        self.update_best_score(model, metrics, preds)
        return metrics

main_scorer = ScorePreds(keep_model=False)

def validate(dataloader, fast=True):
    preds = []
    gt = []

    model.eval()
    with torch.no_grad():
        for (_, batch_data) in tqdm(enumerate(dataloader, 0), disable=True):
            ids = batch_data['ids'].to(device, dtype = torch.long)
            mask = batch_data['mask'].to(device, dtype = torch.long)
            # token_type_ids = batch_data['token_type_ids'].to(device, dtype = torch.long)
            targets = batch_data['targets'].to(device, dtype = torch.long)

            inputs = {"input_ids": ids, "attention_mask": mask, "labels": targets}
            outputs = model(**inputs)

            temp = outputs.logits.cpu().detach().numpy().tolist()
            pred = [max(enumerate(x), key=lambda x: x[1])[0] for x in temp]
            preds.extend(pred)

            labels = inputs['labels']
            gt.extend([gt_label.item() for gt_label in labels])

            # if fast and _ > 20:
            #     break
    return preds, gt

optimizer = None

def get_sizes(data_dir):
    train_path = os.path.join(data_dir, "train.txt")
    labels = {}
    with open(train_path, "r") as infile:
        lines = infile.read().strip().split('\n')

    for line in lines:
        splits = line.split('\t')
        label = splits[-1]
        if label not in labels:
            labels[label] = 0
        labels[label] += 1
    return [labels[x] for x in label_list]

sizes = get_sizes(DATA_PATH)
w = max(sizes) / np.asarray(sizes)
w = torch.from_numpy(w).float().to(device)
w = torch.pow(w, 1)

# loss_fct = CrossEntropyLoss(weight=w)
loss_fct = CrossEntropyLoss()

def train(epochs=1, steps=0):
    if epochs == 0 and steps > 0:
        epochs = 1
    for epoch in range(epochs):
        model.train()
        for idx, batch_data in enumerate(train_dataloader):
            ids = batch_data['ids'].to(device, dtype = torch.long)
            mask = batch_data['mask'].to(device, dtype = torch.long)
            # token_type_ids = batch_data['token_type_ids'].to(device, dtype = torch.long)
            targets = batch_data['targets'].to(device, dtype = torch.long)
            # inputs = {"input_ids": ids, "attention_mask": mask, "labels": targets}
            inputs = {"input_ids": ids, "attention_mask": mask}
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward   backward   optimize
            outputs = model(**inputs)
            loss = loss_fct(outputs.logits.view(-1, len(label_list)), targets.view(-1))

            if idx%5 == 0:
                print(f"Epoch Step: {idx}, Loss:  {loss.item()}", flush=True)

            loss.backward()
            optimizer.step()

            if steps > 0 and idx >= steps:
                break
            if idx > 0 and idx%100 == 0:
                print(main_scorer.calc_score(model, *validate(valid_dataloader, fast=False)), flush=True)
                model.train()
    return epochs

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
TRAIN_EPOCHS = 5

print('\n\nTraining entire model...\n', flush=True)
for i in range(TRAIN_EPOCHS):
    train(1)
    print(f'Epochs = {i + 1}', flush=True)
    preds, gt = validate(valid_dataloader, fast=False)
    metrics = main_scorer.calc_score(model, preds, gt)
    print('VAL SET SCORE', metrics, flush=True)

print('TEST SCORE:', main_scorer.test_score)
