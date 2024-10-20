# Wansi-gemma-2b-instruct-ft-apple-qa
A chatbot trained on a dataset containing questions and answers regarding Apple Devices.

Fine tuning dataset: https://huggingface.co/datasets/Aashi/All_About_Apple_Devices?row=0

# Sample Usage

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Wansi/gemma-2b-instruct-ft-apple-qa")

tokenizer = AutoTokenizer.from_pretrained("Wansi/gemma-2b-instruct-ft-apple-qa")

inputs = tokenizer("How do I take a screenshot on an iPhone?.", return_tensors="pt")
outputs = model.generate(**inputs, max_length= 160)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Output of Sample Usage

How do I take a screenshot on an iPhone?.Sure, here's how to take a screenshot on an iPhone:

1. **Press and hold the power button** on your iPhone.
2. **Slide your finger** across the screen to the left or right edge.
3. A **snapshot icon** will appear. Tap it to take a screenshot.
4. Your screenshot will be saved in the Photos app.

