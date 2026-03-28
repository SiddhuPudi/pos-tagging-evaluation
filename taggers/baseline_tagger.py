def baseline_pos_tag(words):
    predicted_tags = []
    for word in words:
        if word.lower() in ["the", "a", "an"]:
            predicted_tags.append("DT")
        elif word.endswith("ing"):
            predicted_tags.append("VBG")
        elif word.endswith("ed"):
            predicted_tags.append("VBD")
        elif word[0].isupper():
            predicted_tags.append("NNP")
        else:
            predicted_tags.append("NN")
    return predicted_tags