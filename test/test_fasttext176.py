import evidence_features as evf


def test1():
    sentences = [
        "Dieser Satz ist ein Beispiel, aber eher kurz.",
        "Die Kuh macht muh, der Hund wufft aber lauter."
    ]
    encoded = evf.fasttext176_to_int8(sentences)
    assert encoded.shape == (2, 10)
    encoded = evf.fasttext176_to_float(sentences)
    assert encoded.shape == (2, 10)
