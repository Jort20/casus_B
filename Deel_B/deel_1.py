import numpy as np
from collections import defaultdict
import string


def preprocess_text(text):
    
    text = text.lower()
    text = text.strip('\n')
    text = ''.join([char if char in string.ascii_lowercase else ' ' for char in text])
    return text


class NGramModel:
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(lambda: defaultdict(int))  

    def fit(self, text):
        text = preprocess_text(text)
        
        for i in range(len(text) - self.n):
            key = text[i:i + self.n]  
            next_char = text[i + self.n]  
            self.model[key][next_char] += 1  

    def predict_proba(self, key):
        if key not in self.model:
            return {}
        total = sum(self.model[key].values())
        prob_dist = {char: count / total for char, count in self.model[key].items()}
        return prob_dist

    def predict(self, seed, length):
        if len(seed) != self.n:
            raise ValueError(f"Seed must have exactly {self.n} characters.")
        
        result = seed
        for _ in range(length):
            prob_dist = self.predict_proba(seed)
            if not prob_dist:
                break
            next_char = np.random.choice(list(prob_dist.keys()), p=list(prob_dist.values()))
            result += next_char
            seed = result[-self.n:]  
        
        return result


def train_ngram_model(file_path, n):
    
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    
    ngram_model = NGramModel(n)
    
    
    ngram_model.fit(text)
    
    return ngram_model


file_path = '/homes/jrgommers/year 3/Deel_B/kanker-wiki.txt'


ngram_model = train_ngram_model(file_path, 4)


generated_text = ngram_model.predict("kank", 300)
print(generated_text)
