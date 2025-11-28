from nanochat.tokenizer import get_tokenizer

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    token_strings = [tokenizer.decode(token_idx) for token_idx in [range(tokenizer.get_vocab_size() - 1, 0, -1)]]
    print(" ".join(token_strings))