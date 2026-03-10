"""IMDB sentiment classification task."""

import torch
import numpy as np

from nnopt.task import TaskSpec, register_task

_MAX_LEN = 200
_VOCAB_SIZE = 10_000


def _get_imdb_train(subset_size: int):
    def fn():
        from datasets import load_dataset
        ds = load_dataset("imdb", split="train")

        # Simple tokenizer: hash words to vocab indices
        texts, labels = [], []
        for i, row in enumerate(ds):
            if i >= subset_size:
                break
            words = row["text"].lower().split()[:_MAX_LEN]
            ids = [hash(w) % (_VOCAB_SIZE - 1) + 1 for w in words]
            # Pad to MAX_LEN
            ids = ids + [0] * (_MAX_LEN - len(ids))
            texts.append(ids)
            labels.append(row["label"])

        return torch.tensor(texts, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    return fn


def _get_imdb_test(subset_size: int):
    def fn():
        from datasets import load_dataset
        ds = load_dataset("imdb", split="test")

        texts, labels = [], []
        for i, row in enumerate(ds):
            if i >= subset_size:
                break
            words = row["text"].lower().split()[:_MAX_LEN]
            ids = [hash(w) % (_VOCAB_SIZE - 1) + 1 for w in words]
            ids = ids + [0] * (_MAX_LEN - len(ids))
            texts.append(ids)
            labels.append(row["label"])

        return torch.tensor(texts, dtype=torch.long), torch.tensor(labels, dtype=torch.long)
    return fn


register_task(TaskSpec(
    task_id="imdb-50k",
    task_type="sequence",
    dataset_name="IMDB",
    max_params=50_000,
    target_metric=0.80,
    metric_name="accuracy",
    higher_is_better=True,
    max_inference_ms=3.0,
    max_train_time_s=180.0,
    train_subset_size=10_000,
    eval_subset_size=2_000,
    input_shape=(_MAX_LEN,),
    num_classes=2,
    output_description="class logits (N, 2) — 0=negative, 1=positive",
    get_train_data=_get_imdb_train(10_000),
    get_test_data=_get_imdb_test(2_000),
    baseline_metric=0.50,
    tier=2,
    reference_code="""\
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

def solution(train_data, test_data, device):
    X_train, y_train = train_data
    X_train, y_train = X_train.to(device), y_train.to(device)

    class TextCNN(nn.Module):
        def __init__(self, vocab_size=10000, embed_dim=16, max_len=200):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
            self.conv3 = nn.Conv1d(embed_dim, 32, 3, padding=1)
            self.conv5 = nn.Conv1d(embed_dim, 32, 5, padding=2)
            self.fc = nn.Linear(64, 2)
            self.drop = nn.Dropout(0.3)

        def forward(self, x):
            e = self.embed(x).permute(0, 2, 1)  # (B, D, L)
            c3 = torch.relu(self.conv3(e)).max(dim=2).values
            c5 = torch.relu(self.conv5(e)).max(dim=2).values
            cat = torch.cat([c3, c5], dim=1)
            return self.fc(self.drop(cat))

    model = TextCNN().to(device)
    opt = optim.Adam(model.parameters(), lr=0.002)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)

    for epoch in range(15):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            nn.functional.cross_entropy(model(xb), yb).backward()
            opt.step()

    return model
""",
))
