from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel

"""
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-0.5B").to('cuda')
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-0.5B")
"""

model = AutoModelForCausalLM.from_pretrained("./qwen-pubmedqa-final").to('cuda')
tokenizer = AutoTokenizer.from_pretrained("./qwen-pubmedqa-final")

"""
model = PeftModel.from_pretrained(base_model, "./assignment2/new_gpt2-pubmedqa-final").to('cuda')
merged_model = model.merge_and_unload()
"""

def generate_answer(question, context):
    prompt = f"Question: {question}\nContext: {context}\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=512,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )
    
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

test_question = "Which type of lung cancer is afatinib used for? "
test_context = "Clinical perspective of afatinib in non-small cell lung cancer. Reversible ATP-competitive inhibitors targeting the epidermal growth factor receptor (EGFR) have been established as the most effective treatment of patients with advanced non-small cell lung cancer (NSCLC) harboring \"activating\" mutations in exons 19 and 21 of the EGFR gene. However, clinical activity is limited by acquired resistance which on average develops within 10 months of continued treatment. The mechanisms for acquired resistance include selection of the EGFR T790M mutation in approximately 50% of cases, and MET gene amplification, PIK3CA gene mutation, transdifferentiation into small-cell lung cancer and additional rare or unkown mechanisms. Afatinib is a small molecule covalently binding and inhibiting the EGFR, HER2 and HER4 receptor tyrosine kinases. In preclinical studies, afatinib not only inhibited the growth of models with common activating EGFR mutations, but was also active in lung cancer models harboring wild-type EGFR or the EGFR L858R/T790M double mutant. Clinical efficacy of afatinib has been extensively studied in the LUX-Lung study program. These trials showed promising efficacy in patients with EGFR-mutant NSCLC or enriched for clinical benefit from EGFR tyrosine kinase inhibitors gefitinib or erlotinib. Here we review the current status of clinical application of afatinib in NSCLC. We also discuss clinical aspects of resistance to afatinib and strategies for its circumvention."
print(generate_answer(test_question, test_context))