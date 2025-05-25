import torch
import model
from data import Corpus
import torch.nn.functional as F

# load vocabulary
data_loader = Corpus("../data/ptb", batch_size={'train': 1, 'valid': 1}, max_sql=256)
vocab = data_loader.vocabulary
word_to_idx = data_loader.word_id
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

# load model
device = torch.device("cpu")
model = model.LMModel_LSTM(nvoc=len(vocab), dim=256, num_layers=3)
model.load_state_dict(torch.load("model_LSTM.pth", map_location=device))
model = model.to(device)
model.eval()

# input text
input_text = "the meaning of life is"
input_indices = torch.tensor(
    [[word_to_idx.get(word, word_to_idx['<unk>']) for word in input_text.split()]],
    device=device
)

max_len = 50 
temperature = 0.95 # Temperature
top_k = 10  # Top-k
generated_indices = input_indices.squeeze(0).tolist()

with torch.no_grad():
    for _ in range(max_len):
        output = model(input_indices)
        decoded = output[0] if isinstance(output, tuple) else output 
        logits = decoded[:, -1, :] / temperature
        probabilities = F.softmax(logits, dim=-1)

        # remove <unk>
        unk_idx = word_to_idx['<unk>']
        probabilities[0, unk_idx] = 0
        probabilities = probabilities / probabilities.sum()

        # Top-k
        if top_k > 0:
            topk_probs, topk_indices = torch.topk(probabilities, top_k)
            topk_probs = topk_probs / topk_probs.sum()
            next_word_idx = topk_indices[0, torch.multinomial(topk_probs, 1).item()].item()
        else:
            next_word_idx = torch.multinomial(probabilities, num_samples=1).item()

        # penalty
        if next_word_idx in generated_indices[-5:]:
            # 如果最近5个词已出现，强制用概率第二高的词
            sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
            for idx in sorted_indices[0]:
                if idx.item() not in generated_indices[-5:] and idx.item() != unk_idx:
                    next_word_idx = idx.item()
                    break

        print(f"Next word index: {next_word_idx}, word: {idx_to_word.get(next_word_idx, '<unk>')}")
        generated_indices.append(next_word_idx)

        # if next_word_idx == word_to_idx['<eos>']:
        #     break

        input_indices = torch.cat([input_indices, torch.tensor([[next_word_idx]], device=device)], dim=1)

        if input_indices.size(1) > 256:
            input_indices = input_indices[:, -256:]

predicted_words = []
for idx in generated_indices:
    word = idx_to_word.get(idx, '<unk>')
    # if word == '<eos>':
    #     break
    predicted_words.append(word)

sentence = ' '.join(predicted_words)
print("Generated sentence:", sentence)