from kiwipiepy import Kiwi

kiwi = Kiwi()


def kiwi_tokenizer(text):
    return [word[0] for word in kiwi.analyze(text, top_n=1)[0][0]]
