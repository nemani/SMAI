from smai.ngram_model import NgramLanguageModel

ng = NgramLanguageModel('eval','fb', '.',n=3)

ng.load_model()
ng.generate_n_choices()


