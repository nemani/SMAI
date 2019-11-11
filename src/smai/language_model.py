class LanguageModel(object):
    def __init__(self, mode, slug, base_dir):
        self.mode = mode
        self.slug = slug
        self.base_dir = base_dir

    def train(self):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError

    def generateNextWord(self):
        raise NotImplementedError
    
    def setup_for_eval(self):
        raise NotImplementedError
