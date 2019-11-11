from itertools import islice
import collections
import functools
import heapq
import logging
import math
import operator
import pickle
import random
import time
import os

from nltk.util import everygrams
import numpy as np
import pprint

from .language_model import LanguageModel
from .utils import Beam, flatten, sliding_window, tokenize


class NgramLanguageModel(LanguageModel):
    """
    A class representing Language Model
    """

    def __init__(self, mode, slug, base_dir, n=3, use_beam=False, beam_width=10, smoothing_alpha=0.001):
        self.mode = mode
        self.slug = slug
        self.dir = os.path.join(base_dir, "{}_{}".format('NGram', slug) )
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)

        self.n_gram = n
        self.vocabulary = set()
        self.grams_occurence = {}

        self.beam_flag = use_beam
        if self.beam_flag:
            self.beam_width = beam_width
        self.smoothing_alpha = smoothing_alpha

    def get_all_grams(self, token):
        """
        Gives all grams(i.e 1 to n) based on token value
        Input Paramter : token/sentence
        Output Paramter : A list of tuples containing all grams from 1 to n
        """
        if not token:
            return []

        if self.n_gram != 1:
            token = [None]*(self.n_gram-1) + token + [None]*(self.n_gram-1)
        else:
            token = token + [None]

        return list(everygrams(token, max_len=self.n_gram))

    def get_n_gram(self, token):
        """
        Gives n gram from the list of tokens
        Input : A list of tokens
        Output : A list of n-gram tuples
        """
        if not token:
            return []

        if self.n_gram != 1:
            token = [None]*(self.n_gram-1) + token + [None]*(self.n_gram-1)
        else:
            token = token + [None]

        return list(sliding_window(token, self.n_gram))

    def train(self, input_data):
        """
        Trains a n-gram based Language Model based on input text
        Input Paramter : A list of strings [From readlines()]
        """
        token_sequences = []
        for line in input_data:
            token_sequences.append(tokenize(line))

        self.vocabulary = set(flatten(token_sequences))
        self.vocabulary.add(None)

        for item in token_sequences:
            # Iterating over every sentences
            for subitem in self.get_all_grams(item):
                # Iterating over all the grams for a given sentence
                if subitem[:-1] not in self.grams_occurence:
                    # If the n-1 gram word not present in the dictionary
                    self.grams_occurence[subitem[:-1]] = {subitem[-1:][0]: 1}
                    # creating a entry of n-1 word as key and the nth word as the
                    # nested dictionary with value = 1
                else:
                    # example words like "To celebrate this" and "To celebrate that"
                    # it is created as === "to celebrate" : {"this" : 1 , "that" : 1}
                    if subitem[-1:][0] in self.grams_occurence[subitem[:-1]]:
                        # If the key value pair already exist in the nested list
                        self.grams_occurence[subitem[:-1]
                                             ][subitem[-1:][0]] += 1
                    else:
                        # when the outer level key exist but inner level key does not exist
                        self.grams_occurence[subitem[:-1]][subitem[-1:][0]] = 1

        # print("Vocabulary size : ", len(self.vocabulary))
        # print("Tokens in N-grams : ", len(self.grams_occurence))

    def normalize(self, word_counts):
        """
        Normalize the probablility distribution based on counts and smoothen based on alpha-smoothing
        Input : word_count = containing word and their count as key-value pair
        Output : prob_dict = key-value pair as probability distribution
        """
        prob_dict = {}
        total = sum(word_counts.values())

        for key, value in word_counts.items():
            prob_dict[key] = (value+self.smoothing_alpha) / \
                (total+len(self.vocabulary)*self.smoothing_alpha)

        return prob_dict

    def next_word_probability(self, tokens):
        """
        Function returns the most probable next token from the last n-1 gram

        For unigram, it choose a random token and count its probability distribution
        For n-gram n>1, it calculates probability distribution of last n-1 gram 
        Input : A list of Tokens
        Output : A dictionary containing key as the word, and value as the 
        probability for the given ngram/token
        """

        if self.n_gram == 1:
            rand_pair = random.choice(list(self.grams_occurence[()].items()))
            rand_prob = rand_pair[1] / len(self.vocabulary)
            return {rand_pair[0]: rand_prob}
        else:
            # print("Dictionary Keys: ", list(self.grams_occurence.keys()))
            # print("Actual Token : ", tokens)
            for last_n in range(self.n_gram-1, 0, -1):
                token = tuple(tokens[-last_n:])
                print(token)
                if token in self.grams_occurence.keys():
                    return self.normalize(self.grams_occurence[tuple(token)])
                else:
                    return None

    def sample_n_choices(self, distribution, n=1):
        """
        Sample takes a word with distribution of probability and returns the probable word and value

        If beam search is off, it chooses a random choice of probability from the value
        If beam search is on, it chooses top-k number of the key-value and add them in beam list

        Input : A dictionary containing key-value pair of a word and its probability
        Output :
        If beam search is off then returns a tuple containing a (key,value)
        else returns a list of tuples containing key-value pair
        """

        # Handling OOV by randomly selecting a word from the vocabulary
        # print("distribution : ", distribution)

        if distribution is None:
            random_word = random.choice(list(self.grams_occurence.keys()))
            # print("Random Word : ", random_word)
            distribution = self.grams_occurence[random_word]
            # print("distribution : ",distribution)

        beam_list = set()
        counter = 0

        k = np.random.choice(list(distribution.keys()), n, list(distribution.values()))
        
        for each in list(k):
            v = distribution[each]
            beam_list.add((each,v))
        
        if not self.beam_flag:
            return list(beam_list)
        else:
            # we take at max 5 word for our case
            while counter <= 5:
                beam_list.add((k, v))
                k = np.random.choice(
                    list(distribution.keys()), n, list(distribution.values()))
                v = distribution[k]
                if len(beam_list) == self.beam_width:
                    break
                counter = counter + 1
            #print("Beam List : ", list(beam_list))

            return list(beam_list)

    def beam_search(self, token, beam_width=10, clip_len=20):
        """
         Search and return the most possible beam based on the current token
         Input:
         token = list of tokens
         beam_width = Beam Size
         clip_len = Clip length to check the most possible lowest value of beam size

         Output :
         Most probable sentence with its probability and not including the SOS token.

        """
        prev_beam = Beam(beam_width)  # Creating an initial beam(i.e heap)
        prev_beam.add(1.0, False, token)  # Add a starting value to the beam
        while True:
            curr_beam = Beam(beam_width)
            for (prob, complete, prefix_token) in prev_beam:
                if complete == True:
                    if len(prefix_token) >= clip_len:
                        curr_beam.add(prob, True, prefix_token)
                else:
                    for next_word, next_prob in self.sample_n_choices(self.next_word_probability(prefix_token)):
                        if next_word == None:
                            if len(prefix_token) >= clip_len:
                                curr_beam.add(prob*next_prob,
                                              True, prefix_token)
                        else:
                            curr_beam.add(prob*next_prob, False,
                                          prefix_token + [next_word])

            (best_prob, complete, best_token) = max(curr_beam)

            if complete == True and len(best_token) >= clip_len:
                return (best_token[1:], best_prob)
            prev_beam = curr_beam

    def generate_n_choices(self, n=1, seed_text=None):
        """
        Function generates sentences based on trained model
        Input : seed_text = If seed_text is True then look for seed text
        Output : Return a sentence token in the form of a list
        """
        tokens_size = self.n_gram - 1
        tokens = []

        if seed_text:
            tokens_size = len(tokens)
            tokens = seed_text.lower().split(' ')

        if self.beam_flag:
            #print("Tokens before Beam : ", tokens)
            tokens.extend(self.beam_search(tokens, self.beam_width)[0])
            #print("Tokens After beam : ", tokens)
            del tokens[1:tokens_size]  # Deleting 2nd word since they repeat
        else:
            while True:
                choices = self.sample_n_choices(self.next_word_probability(tokens), n=n)
                choice = choices[0][0]
                next_word = choice
                if next_word is None:
                    break
                tokens.append(next_word)

        return tokens

    def n_gram_probability(self, ngram_word, next_word_in_ngram):
        """
        Probability counting for perplexity.
        Input : ngram_word is a tuple to match with the count dictionary keys
                        next_word_in_ngram is the word to match up with the specific ngram
        Output : probability of the specific (ngram_word, next_word_in_ngram)
        """
        if ngram_word in self.grams_occurence:
            if next_word_in_ngram in self.grams_occurence[ngram_word]:
                occurence_count = self.grams_occurence[ngram_word][next_word_in_ngram]
                total = sum(self.grams_occurence[ngram_word].values())
            else:
                occurence_count = 1
                total = sum(self.grams_occurence[ngram_word].values()) + 1

            probability = (occurence_count + self.smoothing_alpha) / \
                (total + (len(self.vocabulary) * self.smoothing_alpha))

        else:
            occurence_count = 0
            total = 1
            probability = random.random() / (len(self.vocabulary))

        return probability

    def perplexity(self, test_data):
        """
        Calculates Perplexity for the test data.

        If ngram is matched in the training data, it adds the probability distribution on perplexity
        If ngram is not matched, <UNK> is considered and a random probability distribution is added on 
        perplexity.

        Input : A list of tokens of Test Set

        Output : Perplexity Value
        """
        perplexity_value = 0
        num = 0
        for item in test_data:
            # if i < 10:
            # 	print(item)
            # 	i = i + 1
            for subitem in self.get_all_grams(item):
                #print("Length of subitem : ", len(subitem))
                num = num + 1
                perplexity_value += math.log(
                    self.n_gram_probability(subitem[:-1], subitem[-1:][0]), 2)

        perplexity_value = math.pow(2, -1 * (perplexity_value/num))
        return perplexity_value

    def save_model(self):
        with open(os.path.join(self.dir, 'NgramModel.pkl'), 'wb') as f:
            pickle.dump(self.__dict__, f)
        # with open('vocabulary.pkl', 'wb') as f:
        #     pickle.dump(self.vocabulary, f)

        with open(os.path.join(self.dir, 'grams.pkl'), 'wb') as f:
            pickle.dump(self.grams_occurence, f)

    def load_model(self):
        curr_mode = self.mode
        with open(os.path.join(self.dir,'NgramModel.pkl'), 'rb') as f:
            attrs = pickle.load(f)
            self.__dict__.update(attrs)
            self.mode = curr_mode

        # with open('vocabulary.pkl', 'rb') as f:
        #     self.vocabulary = pickle.load(f)

        with open(os.path.join(self.dir, 'grams.pkl'), 'rb') as f:
            self.grams_occurence = pickle.load(f)
