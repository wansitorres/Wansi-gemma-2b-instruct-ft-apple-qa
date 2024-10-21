# Wansi-gemma-2b-instruct-ft-apple-qa
A chatbot trained on gemma2b-it fine tuned on a dataset containing questions and answers regarding Apple Devices.

Fine tuning dataset: https://huggingface.co/datasets/Aashi/All_About_Apple_Devices?row=0

# Sample Usage

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Wansi/gemma-2b-instruct-ft-apple-qa")

tokenizer = AutoTokenizer.from_pretrained("Wansi/gemma-2b-instruct-ft-apple-qa")

inputs = tokenizer(""How do I safeguard my iPhone from potential damage caused by exposure to liquids, dust, or extreme temperatures?", return_tensors="pt")

outputs = model.generate(**inputs, max_length= 160)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# Output of Sample Usage

Use protective cases like tempered glass or a case that shields the edges of your iPhone.  
 Avoid touching sensitive areas like the screen, buttons, or charging port. 
 Store your iPhone in a clean, dry place to reduce the risk of damage.  modelKeep your iPhone dry with a case or waterproof sleeve.  
 Avoid using your iPhone in extreme temperatures or in humid environments.  
 Clean your iPhone regularly with a soft, dry cloth.  
 Avoid dropping or knocking your iPhone.  
 Do not expose your iPhone to extreme temperatures, such as hot, cold, or extreme humidity.  
 Consult an Apple Support professional if you have concerns.

