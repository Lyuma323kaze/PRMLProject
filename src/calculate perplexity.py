import torch
import math
import model
from data import Corpus
import torch.nn.functional as F
import pickle

# load model and data
def load_model_and_data(model_path, corpus_path, model_class, nvoc, dim, num_layers, device):
    with open("word_to_idx.pkl", "rb") as f:
        word_to_idx = pickle.load(f)
    with open("vocab.pkl", "rb") as f:
        vocab = pickle.load(f)

    data_loader = Corpus(
        corpus_path,
        batch_size={'train': 1, 'valid': 10},
        max_sql=256,
        word_id=word_to_idx,     
        vocabulary=vocab
    )
    nvoc = len(word_to_idx)
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    model_instance = model_class(nvoc=nvoc, dim=dim, num_layers=num_layers)
    checkpoint = torch.load(model_path, map_location=device)
    model_instance.load_state_dict(checkpoint['model_state_dict'])
    model_instance = model_instance.to(device)
    model_instance.eval()

    return model_instance, data_loader, vocab, word_to_idx, idx_to_word

# Calculate Perplexity
def calculate_perplexity(model_, data_loader, device):
    data_loader.set_valid()
    data_, target, end_flag = data_loader.get_batch()
    model_.eval()
    total_loss = 0
    total_tokens = 0
    print(f"Calculating Perplexity (ignore <unk>)...")
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
    unk_idx = data_loader.word_id['<unk>']

    while not end_flag:
        with torch.no_grad():
            data_, target, end_flag = data_loader.get_batch()
            data_ = data_.to(device)
            target = target.to(device)
            decode = model_(data_)[0]
            logits = decode.view(-1, decode.size(-1))
            targets = target.view(-1)
            loss = criterion(logits, targets)  # shape: (N,)
            mask = (targets != unk_idx)
            total_loss += (loss * mask).sum().item()
            total_tokens += mask.sum().item()
    if total_tokens == 0:
        print("No valid tokens (excluding <unk>) found!")
        return float('inf')
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    print(f"Perplexity (ignore <unk>): {perplexity}")
    return perplexity

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lstm_model_path = "lstm_checkpoint.pth"
    rnn_model_path = "rnn_checkpoint.pth"
    transformer_model_path = "transformer_checkpoint.pth"
    new_corpus_path = "../data/ptb/Tiny_Shakespeare"

    nvoc = 10000 
    dim = 256
    num_layers_lstm = 3
    num_layers_rnn = 3
    num_layers_transformer = 6

    # LSTM
    model_LSTM, new_data_loader, _, _, _ = load_model_and_data(
        lstm_model_path, new_corpus_path, model.LMModel_LSTM, nvoc, dim, num_layers_lstm, device
    )
    lstm_perplexity = calculate_perplexity(model_LSTM, new_data_loader, device)

    #RNN
    model_RNN, _, _, _, _ = load_model_and_data(
        rnn_model_path, new_corpus_path, model.LMModel_RNN, nvoc, dim, num_layers_rnn, device
    )
    rnn_perplexity = calculate_perplexity(model_RNN, new_data_loader, device)

    #Transformer
    model_transformer, _, _, _, _ = load_model_and_data(
        transformer_model_path, new_corpus_path, model.LMModel_transformer, nvoc, dim, num_layers_transformer, device
    )
    transformer_perplexity = calculate_perplexity(model_transformer, new_data_loader, device)

    # save perplexity results
    with open("new_data_perplexity.txt", "w") as f:
        f.write(f"LSTM Perplexity: {lstm_perplexity}\n")
        f.write(f"RNN Perplexity: {rnn_perplexity}\n")
        f.write(f"Transformer Perplexity: {transformer_perplexity}\n")

    print("Perplexity results saved to new_data_perplexity.txt")