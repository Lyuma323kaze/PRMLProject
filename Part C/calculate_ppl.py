from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from tqdm import tqdm 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_path = "./qwen-pubmedqa-final"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.eval()

dataset = load_dataset("kroshan/BioASQ", split="validation")
#dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
def format_bioasq(example):
    """ 将问答数据转换为模型输入格式 """
    return f"Question: {example['question']}\nAnswer: {example['text']}"
texts = [format_bioasq(example) for example in dataset]

max_length = tokenizer.model_max_length
stride = 512
total_logits = []
total_loss = []


for text in tqdm(texts, desc="Calculating Perplexity", unit="text"):
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encodings.input_ids.to(device)
    
    total_chunks = (input_ids.size(1) - max_length) // stride + 1
    for i in tqdm(range(0, input_ids.size(1), stride), 
                desc="Processing chunks", 
                leave=False,
                total=total_chunks):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        trg_len = end_loc - begin_loc
        
        input_ids_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_ids_chunk.clone()
        target_ids[:, :-trg_len] = -100
        
        with torch.no_grad():
            outputs = model(input_ids_chunk, labels=target_ids)
            loss = outputs.loss
        
        total_loss.append(loss.item())

perplexity = torch.exp(torch.tensor(sum(total_loss) / len(total_loss))).item()
print(f"Perplexity: {perplexity}")
