import stanza
nlp = stanza.Pipeline(lang="en", processors='tokenize,pos', use_gpu=False)

def stanza_pos_tag(words):
    sentence = " ".join(words)
    doc = nlp(sentence)
    predicted_tags = []
    stanza_words = []
    stanza_tags = []
    for sent in doc.sentences:
        for word in sent.words:
            stanza_words.append(word.text)
            stanza_tags.append(word.xpos)
    i = 0
    j = 0
    while i < len(words) and j < len(stanza_words):
        if words[i] == stanza_words[j]:
            predicted_tags.append(stanza_tags[j])
            i += 1
            j += 1
        else:
            j += 1
    return predicted_tags