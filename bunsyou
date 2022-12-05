
!pip install transformers==4.4.2

!pip install sentencepiece==0.1.91
tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-medium")
model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-medium")

input = tokenizer.encode("昔々あるところに", return_tensors="pt")
output = model.generate(input, do_sample=True, max_length=140, num_return_sequences=1)
print(tokenizer.batch_decode(output))
