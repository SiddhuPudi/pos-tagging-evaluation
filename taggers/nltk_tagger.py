import nltk

def nltk_pos_tag(words):
    tagged = nltk.pos_tag(words)
    predicted_tags = [tag for word, tag in tagged]
    return predicted_tags