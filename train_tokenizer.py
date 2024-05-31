import os
import time
from tokenizer.regex import RegexTokenizer

# open some text and train a vocab of 512 tokens
text = open("tests/tinystories.txt", "r", encoding="utf-8").read()

t0 = time.time()

# construct the Tokenizer object and kick off verbose training
tokenizer = RegexTokenizer()
tokenizer.train(text, 10000, verbose=True)
# writes two files in the models directory: name.model, and name.vocab
prefix = "trained_tokenizer"
tokenizer.save(prefix)
t1 = time.time()

print(f"Training took {t1 - t0:.2f} seconds")