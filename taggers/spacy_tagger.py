import spacy
from spacy.tokens import Doc

nlp = spacy.load("en_core_web_sm")

def spacy_pos_tag(words):
    sentence = " ".join(words)
    doc = nlp(sentence)
    predicted_tags = []
    doc_words = [token.text for token in doc]
    i = 0
    j = 0
    while i < len(words) and j < len(doc):
        if words[i] == doc[j].text:
            predicted_tags.append(doc[j].tag_)
            i += 1
            j += 1
        else:
            j += 1
    return predicted_tags