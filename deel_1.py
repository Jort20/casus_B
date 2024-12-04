import numpy as np
from collections import defaultdict

"""
Model to predict text based on a corpus in a mediate innocent way based on the possible outcome character on a n length part of the corpus.
"""
class NGramModel:
    def __init__(self, n):
        self.n = n
        self.model = defaultdict(lambda: defaultdict(int))  

    """
    Make a dict with the n length part of the corpus(text), and the following letter.
    input: text, a String containg the corpus.
    """
    def fit(self, text):        
        for i in range(len(text) - self.n):
            key = text[i:i + self.n]  
            next_char = text[i + self.n]  
            self.model[key][next_char] += 1  

    """
    Based on the dict made in the function fit. The possible outputs will be returned based on a seed with length n.
    input: key, a string containing the seed.
    return: prob_dist, a dict with the probability of a certain output with the seed.
    """
    def predict_proba(self, key):
        if key not in self.model:
            return {}
        #calculate the probability of every possible outcome
        total = sum(self.model[key].values())
        prob_dist = {char: count / total for char, count in self.model[key].items()}
        return prob_dist
    
    """ 
    Predict a scentence with length m, based on a seed with length n using the dict made in the function fit.
    input: seed, a string containg the start of the next scentence.
           length, a integer with the expected length of the predicted scentence.
    return: result, a string containg the predicted scentence.
    """
    def predict(self, seed, length):
        if len(seed) != self.n:
            raise ValueError(f"Seed must have exactly {self.n} characters.")
        
        result = seed
        for _ in range(length):
            #predict probabilities for the given seed
            prob_dist = self.predict_proba(seed)
            if not prob_dist:
                break
            #determine the next char based on the seed
            next_char = np.random.choice(list(prob_dist.keys()), p=list(prob_dist.values()))
            result += next_char
            #change seed to include the new char
            seed = result[-self.n:]  
        
        return result

