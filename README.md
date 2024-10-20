# Wansi-gemma-2b-instruct-ft-apple-qa

Fine tuning dataset: https://huggingface.co/datasets/Aashi/All_About_Apple_Devices?row=0

# Sample Usage

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Wansi/gemma-2b-instruct-ft-apple-qa")
tokenizer = AutoTokenizer.from_pretrained("Wansi/gemma-2b-instruct-ft-apple-qa")

inputs = tokenizer("How do I take a screenshot on an iPhone?.", return_tensors="pt")
outputs = model.generate(**inputs, max_length= 160)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

