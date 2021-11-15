from transformers import GPT2Tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')
tokenizer.pad_token = tokenizer.eos_token
