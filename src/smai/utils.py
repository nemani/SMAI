import heapq
import string
import re
import sys


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def tokenize(text):
    """
    Tokenize text based on several parameters using Reg expression 
    (punctuation, capitalization, contraction etc)

    Parameters
    ----------
    text: string
            Takes each lines from the train data

    Return
    ------
    list
            Tokenized string in a list form
    """
    text = text.replace('--', ' ')
    text = text.replace('  ', ' ')
    text = text.replace('!!', '!')

    text = clean_regex(text)
    #punctuation_table = dict((ord(char), None) for char in string.punctuation)
    #text = text.translate(punctuation_table)

    # remove certain punctuation chars
    text = re.sub("[()]", "", text)
    # collapse multiples of certain chars
    text = re.sub('([.-])+', r'\1', text)

    # Removing URL's from the text
    text = re.sub(
        r"(https?:\/\/)(\s)*(www\.)?(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*", r"", text)

    # Removing URL's starting with www
    text = re.sub(
        r"(www\.)(\s)*((\w|\s)+\.)*([\w\-\s]+\/)*([\w\-]+)((\?)?[\w\s]*=\s*[\w\%&]*)*", r"", text)

    if text[-1] in ".,!?":
        text = text[:-1] + " " + text[-1]
    
    # pad sentence punctuation chars with whitespace
    text = re.sub('([^0-9])([.,!?"$\'])([^0-9])', r'\1 \2 \3', text)
    
    tokens = text.split()
    
    tokens = [word.lower() for word in tokens if word is not ""]
    return tokens


def clean_regex(text):
    """
    Clean contraction from the text

    Parameters
    ----------
    text: string
            Text lines of string

    Return
    ------
    string
            Cleaned contraction
    """
    clean_dict = {
        r"i'm": r"i am",
        r"he's": r"he is",
        r"she's": r"she is",
        r"that's": r"that is",
        r"what's": r"what is",
        r"where's": r"where is",
        r"\'ll": r" will",
        r"\'ve": r" have",
        r"\'re": r" are",
        r"\'d": r" would",
        r"won't": r"will not",
        r"can't": r"can not",
        r"I've": r"I have"
    }
    for each in clean_dict:
        text = re.sub(each, clean_dict[each], text)

    return text


def detokenize(tokens):
    """
    Detokenize list of tokens based on parameters using regex (punctuation, capitalization,
    padding etc)

    Parameters
    ----------
    tokens: list
            list of tokens generated

    Return
    ------
    string
            As a form of text
    """
    tokens = list(filter(None.__ne__, tokens))
    text = ' '.join(tokens)
    # correct whitespace padding around punctuation
    text = re.sub('\\s+([.,!?])\\s*', r'\1 ', text)
    # capitalize first letter
    text = text.capitalize()
    # capitalize letters following terminated sentences
    text = re.sub('([.!?]\\s+[a-z])', lambda c: c.group(1).upper(), text)
    return text


def flatten(token):
    """
    Converts a nested list into a 1-dimensional list
    Input : A nested List containing all the s entences
    Output : A 1 dimensional list conaining all the words
    """
    for i in token:
        if isinstance(i, (list, tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i


def sliding_window(sequence, window):
    """
    Function returns iterating each element as sliding window
    Input : A list of tokens and window size
    Output : A list of n-gram tuples
    """

    it = iter(sequence)
    result = tuple(islice(it, window))
    if len(result) == window:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


class Beam(object):
    """
    A class to represent a Beam data structure

    ...
    Attributes
    ----------

    heap: list
            list containing probability, token and setence completetion boolean information

    beam_width: int
            size of beam
    """
    # For comparison of prefixes, the tuple (prefix_probability, complete_sentence) is used.
    # This is so that if two prefixes have equal probabilities then a complete sentence is preferred over an incomplete one since (0.5, False) < (0.5, True)

    def __init__(self, beam_width):
        """
        Initiate the beam

        Parameters
        ----------
        beam_width: int
                size of beam
        """
        self.heap = list()
        self.beam_width = beam_width

    def add(self, prob, complete, prefix):
        """
        Add item to the heap queue. Also pop from heap if beam size is greate than max defined

        Parameters
        ----------
        prob: float
                Probability value of current sentence in the beam
        complete: boolean
                if the sentence is completed or not
        prefix:
                last n-1 token in the current sentence
        """

        heapq.heappush(self.heap, (prob, complete, prefix))
        if len(self.heap) > self.beam_width:
            heapq.heappop(self.heap)

    def __iter__(self):
        """
        Iterate over the heap structure
        """
        return iter(self.heap)
