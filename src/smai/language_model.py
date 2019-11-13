from prompt_toolkit import prompt
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.application.current import get_app
from prompt_toolkit.filters import Condition
from prompt_toolkit.filters.utils import to_filter
from prompt_toolkit.key_binding import KeyBindings

kb = KeyBindings()

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

    def start_prompt(self):
        self.show_completions = False
        f = to_filter(Condition(lambda: self.show_completions))
        prompt('>>> ', multiline=True, completer=MyCustomCompleter(self))


# @kb.add('c-space')
# def _(event):
#     " Initialize autocompletion, or select the next completion. "
#     buff = event.app.current_buffer
#     print(buff)
#     if buff.complete_state:
#         buff.complete_next()
#     else:
#         buff.start_completion(select_first=False)


class MyCustomCompleter(Completer):
    def __init__(self, LM):
        self.LM = LM

    def get_completions(self, document, complete_event):
        next_options = self.LM.generate_n_choices(n=3, seed_text=document.text)
        
        for each in next_options:
            yield Completion(each, start_position=0)
