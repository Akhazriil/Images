import re
from collections import defaultdict

REGULAR = r"\b[а-яА-ЯёЁ]+\b"

with open("data/MasterAndMargarita.txt", "r", encoding="utf-8") as file:
    text = file.read()
    words = re.findall(REGULAR, text)
    filtered_array = []
    for word in words:
        if len(word) > 1:
            filtered_array.append(word)

words_count_dict = defaultdict(int)
for word in filtered_array:
    word = word.lower()
    words_count_dict[word] += 1
    if len(words_count_dict) > 2500:
        break

vocab = {}
for word, freq in words_count_dict.items():
    vocab[' '.join(list(word)) + ' </w>'] = freq

def get_stats(vocab):
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    new_vocab = {}
    bigram = ' '.join(pair)
    merged = ''.join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, merged)
        new_vocab[new_word] = freq
    return new_vocab

num_merges = 50
merges = []

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best_pair = max(pairs, key=pairs.get)
    if pairs[best_pair] < 2:
        break
    vocab = merge_vocab(best_pair, vocab)
    merges.append(best_pair)

tokens = set()
for word in vocab.keys():
    tokens.update(word.split())

token_to_id = {token: idx for idx, token in enumerate(tokens)}
id_to_token = {idx: token for token, idx in token_to_id.items()}

test_text = "Мастер и Маргарита"
test_words = re.findall(REGULAR, test_text)
final_tokens = []

for word in test_words:
    word_tokens = list(word) + ['</w>']
    for pair in merges:
        merged = ''.join(pair)
        i = 0
        new_tokens = []
        while i < len(word_tokens):
            if i < len(word_tokens) - 1 and word_tokens[i] + word_tokens[i+1] == merged:
                new_tokens.append(merged)
                i += 2
            else:
                new_tokens.append(word_tokens[i])
                i += 1
        word_tokens = new_tokens
    final_tokens.extend(word_tokens)

ids = []
for t in final_tokens:
    if t in token_to_id:
        ids.append(token_to_id[t])
    else:
        ids.append(-1)

decoded = ''
for i in ids:
    if i != -1:
        decoded += id_to_token[i]
    else:
        decoded += '<UNK>'
decoded = decoded.replace('</w>', ' ')