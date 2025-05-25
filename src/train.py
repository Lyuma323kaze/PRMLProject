# coding: utf-8
import argparse
import time
import math
import torch
import torch.optim as optim
import torch.nn as nn

import data
import model
import os
import os.path as osp

'''
ATTENTION!
working file: ./src
'''

# os.environ["TORCH_USE_CUDA_DSA"] = "1"
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


parser = argparse.ArgumentParser(description='PyTorch ptb Language Model')
parser.add_argument('--epochs', type=int, default=50,
                    help='upper epoch limit')
parser.add_argument('--train_batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--eval_batch_size', type=int, default=10, metavar='N',
                    help='eval batch size')
parser.add_argument('--max_sql', type=int, default=256,
                    help='sequence length')
parser.add_argument('--num_layers_rnn', type=int, default=3, help='RNN层数')
parser.add_argument('--num_layers_lstm', type=int, default=3, help='LSTM层数')
parser.add_argument('--num_layers_transformer', type=int, default=6, help='Transformer层数')
parser.add_argument('--seed', type=int, default=1234,
                    help='set random seed')
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--use_pe', action="store_true")
parser.add_argument('--cuda', action='store_true', help='use CUDA device')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU device id used')
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

# Use gpu or cpu to train
use_gpu = True

if use_gpu:
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(args.gpu_id)
else:
    device = torch.device("cpu")

# load data
train_batch_size = args.train_batch_size
eval_batch_size = args.eval_batch_size
batch_size = {'train': train_batch_size,'valid':eval_batch_size}
data_loader = data.Corpus("../data/ptb", batch_size, args.max_sql)



########################################
# Build LMModel model (build your language model here)
# transformer
model_transformer = model.LMModel_transformer(
    nvoc=len(data_loader.vocabulary),
    num_layers=args.num_layers_transformer,
    dim=args.emb_dim,
    nhead=args.num_heads
)
model_transformer = model_transformer.to(device)
optimizer_transformer = optim.Adam(model_transformer.parameters(), lr=1e-3)
# RNN
model_RNN = model.LMModel_RNN(
    nvoc=len(data_loader.vocabulary),
    dim=args.emb_dim,
    num_layers=args.num_layers_rnn
)
model_RNN = model_RNN.to(device)
optimizer_RNN = optim.Adam(model_RNN.parameters(), lr=1e-3)
# LSTM
model_LSTM = model.LMModel_LSTM(
    nvoc=len(data_loader.vocabulary),
    dim=args.emb_dim,
    num_layers=args.num_layers_lstm
)
model_LSTM = model_LSTM.to(device)
optimizer_LSTM = optim.Adam(model_LSTM.parameters(), lr=1e-3)

criterion = nn.CrossEntropyLoss()

# Evaluation Function
# Calculate the average cross-entropy loss between the prediction and the ground truth word.
# And then exp(average cross-entropy loss) is perplexity.

def evaluate(model_):
    data_loader.set_valid()
    data_, target, end_flag = data_loader.get_batch()
    model_.eval()
    total_loss = 0
    total_tokens = 0
    unk_idx = data_loader.word_id['<unk>']
    print(f"Validating")
    while not end_flag:
        with torch.no_grad():
            data_, target, end_flag = data_loader.get_batch()
            data_ = data_.to(device)
            target = target.to(device)
            decode = model_(data_)[0]
            logits = decode.view(-1, decode.size(-1))
            targets = target.view(-1)
            loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
            mask = (targets != unk_idx)
            total_loss += (loss * mask).sum().item()
            total_tokens += mask.sum().item()
    if total_tokens == 0:
        return float('inf')
    avg_loss = total_loss / total_tokens
    print(f"The average loss (ignore <unk>) is {avg_loss}")
    return math.exp(avg_loss)


# Train Function
def train(model_, optimizer_):
    torch.autograd.set_detect_anomaly(True)
    data_loader.set_train()
    data_, target, end_flag = data_loader.get_batch()
    model_.train()
    total_loss = 0
    total_tokens = 0
    unk_idx = data_loader.word_id['<unk>']
    while not end_flag:
        data_, target, end_flag = data_loader.get_batch()
        data_ = data_.to(device)
        target = target.to(device)
        decode = model_(data_)[0]
        logits = decode.view(-1, decode.size(-1))
        targets = target.view(-1)
        loss = nn.CrossEntropyLoss(reduction='none')(logits, targets)
        mask = (targets != unk_idx)
        loss_sum = (loss * mask).sum()
        optimizer_.zero_grad()
        loss_sum.backward()
        optimizer_.step()
        total_loss += loss_sum.item()
        total_tokens += mask.sum().item()
    if total_tokens == 0:
        return float('inf')
    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)


# Loop over epochs for transformer
train_perplexity_transformer = []
valid_perplexity_transformer = []
train_perplexity_RNN = []
valid_perplexity_RNN = []
train_perplexity_LSTM = []
valid_perplexity_LSTM = []

def see_epoch(model_, train_ls, valid_ls, name: str = None, optimizer_ = None, start_epoch=1):
    for epoch in range(start_epoch, args.epochs + 1):
        print(f"Start training epoch ({name}) {epoch}")
        train_ls.append(train(model_, optimizer_))
        valid_ls.append(evaluate(model_))
        # 保存模型和优化器状态
        torch.save({
            'epoch': epoch,
            'model_state_dict': model_.state_dict(),
            'optimizer_state_dict': optimizer_.state_dict(),
        }, f"{name.lower()}_checkpoint.pth")
        print(f"Checkpoint saved for {name} at epoch {epoch}")
    print(f"Train Perplexity {name}: {train_ls}")
    print(f"Valid Perplexity {name}: {valid_ls}")

# 加载模型和优化器状态
start_epoch_LSTM = 1
if os.path.exists("lstm_checkpoint.pth"):
    checkpoint = torch.load("lstm_checkpoint.pth", map_location=device)
    model_LSTM.load_state_dict(checkpoint['model_state_dict'])
    optimizer_LSTM.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch_LSTM = checkpoint['epoch'] + 1
    print(f"Resuming LSTM training from epoch {start_epoch_LSTM}")

start_epoch_RNN = 1
if os.path.exists("rnn_checkpoint.pth"):
    checkpoint = torch.load("rnn_checkpoint.pth", map_location=device)
    model_RNN.load_state_dict(checkpoint['model_state_dict'])
    optimizer_RNN.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch_RNN = checkpoint['epoch'] + 1
    print(f"Resuming RNN training from epoch {start_epoch_RNN}")

start_epoch_transformer = 1
if os.path.exists("transformer_checkpoint.pth"):
    checkpoint = torch.load("transformer_checkpoint.pth", map_location=device)
    model_transformer.load_state_dict(checkpoint['model_state_dict'])
    optimizer_transformer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch_transformer = checkpoint['epoch'] + 1
    print(f"Resuming Transformer training from epoch {start_epoch_transformer}")

see_epoch(model_LSTM, train_perplexity_LSTM, valid_perplexity_LSTM, "LSTM", optimizer_LSTM, start_epoch=start_epoch_LSTM)
torch.save(model_LSTM.state_dict(), "model_LSTM.pth")  # 保存LSTM模型参数

see_epoch(model_RNN, train_perplexity_RNN, valid_perplexity_RNN, "RNN", optimizer_RNN, start_epoch=start_epoch_RNN)
torch.save(model_RNN.state_dict(), "model_RNN.pth")    # 保存RNN模型参数

see_epoch(model_transformer, train_perplexity_transformer, valid_perplexity_transformer, "Transformer", optimizer_transformer, start_epoch=start_epoch_transformer)
torch.save(model_transformer.state_dict(), "model_transformer.pth")  # 保存Transformer模型参数
# 统一保存 perplexity 到文件
with open("train_perplexity.txt", "w") as f:
    f.write("LSTM: " + str(train_perplexity_LSTM) + "\n")
    f.write("RNN: " + str(train_perplexity_RNN) + "\n")
    f.write("Transformer: " + str(train_perplexity_transformer) + "\n")

with open("valid_perplexity.txt", "w") as f:
    f.write("LSTM: " + str(valid_perplexity_LSTM) + "\n")
    f.write("RNN: " + str(valid_perplexity_RNN) + "\n")
    f.write("Transformer: " + str(valid_perplexity_transformer) + "\n")


import pickle
with open("word_to_idx.pkl", "wb") as f:
    pickle.dump(data_loader.word_id, f)
with open("vocab.pkl", "wb") as f:
    pickle.dump(data_loader.vocabulary, f)

print("Perplexity saved to train_perplexity.txt and valid_perplexity.txt")