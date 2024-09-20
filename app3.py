from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("fine_tuned_aragpt2")
tokenizer = GPT2Tokenizer.from_pretrained("fine_tuned_aragpt2")
