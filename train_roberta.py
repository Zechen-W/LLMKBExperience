import random
import os
import numpy as np
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoConfig
from transformers import RobertaPreTrainedModel, RobertaModel
from transformers import AdamW, get_scheduler
from tqdm.auto import tqdm

import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

checkpoint = (
    "/home/wzc2022/dgt_workspace/LLM-Knowledge-alignment-dgt/roberta_weights/roberta-base"
)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def seed_everything(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


class RobertaDataset(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        Data = {}
        with open(data_file, "r") as f:
            lines = f.readlines()
            idx = 0
            for line in lines:
                sample = json.loads(line.strip())
                if sample["score"].isnumeric():
                    Data[idx] = sample
                    idx += 1
        return Data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collote_fn(batch_samples):
    batch_question, batch_ctx = [], []
    batch_label = []
    for sample in batch_samples:
        batch_question.append(sample["question"])
        batch_ctx.append(sample["ctx"])
        batch_label.append(int(sample["score"]))
    X = tokenizer(
        batch_question,
        batch_ctx,
        padding=True,
        truncation=True,
        return_tensors="pt",
    ).to(device)
    y = torch.tensor(batch_label).to(device)
    return X, y


class RobertaCLS(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(768, 6)
        self.post_init()

    def forward(self, x):
        roberta_output = self.roberta(**x)
        cls_vectors = roberta_output.last_hidden_state[:, 0, :]
        cls_vectors = self.dropout(cls_vectors)
        logits = self.classifier(cls_vectors)
        return logits


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f"loss: {0:>7f}")
    finish_step_num = (epoch - 1) * len(dataloader)

    model.train()
    for step, (X, y) in enumerate(dataloader, start=1):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        progress_bar.set_description(f"loss: {total_loss/(finish_step_num + step):>7f}")
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader, model, mode="Test"):
    assert mode in ["Valid", "Test"]
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    correct /= size
    print(f"{mode} Accuracy: {(100*correct):>0.1f}%\n")
    return correct


def main():
    seed_everything(42)
    learning_rate = 1e-5
    batch_size = 4
    epoch_num = 10

    train_data = RobertaDataset(
        "/home/wzc2022/dgt_workspace/LLM-Knowledge-alignment-dgt/data/part_qa_pair_scores_5.jsonl"
    )
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collote_fn
    )

    valid_data = RobertaDataset(
        "/home/wzc2022/dgt_workspace/LLM-Knowledge-alignment-dgt/data/part_qa_pair_scores_1.jsonl"
    )
    valid_dataloader = DataLoader(
        valid_data, batch_size=batch_size, shuffle=False, collate_fn=collote_fn
    )

    config = AutoConfig.from_pretrained(checkpoint)
    model = RobertaCLS.from_pretrained(checkpoint, config=config).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch_num * len(train_dataloader),
    )

    total_loss = 0.0
    best_acc = 0.0
    for t in range(epoch_num):
        print(f"Epoch {t+1}/{epoch_num}\n-------------------------------")
        total_loss = train_loop(
            train_dataloader, model, loss_fn, optimizer, lr_scheduler, t + 1, total_loss
        )
        valid_acc = test_loop(valid_dataloader, model, mode="Valid")
        if valid_acc > best_acc:
            best_acc = valid_acc
            print("saving new weights...\n")
            torch.save(
                model.state_dict(),
                f"./roberta_weights/epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin",
            )
    print("Done!")


if __name__ == "__main__":
    main()
