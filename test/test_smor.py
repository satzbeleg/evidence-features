import evidence_features as evf


def test1():
    sentences = [
        "Dieser Satz ist ein Beispiel, aber eher kurz.",
        "Die Kuh macht muh, der Hund wufft aber lauter."
    ]
    enc = evf.smor_to_int8(sentences)
    assert enc.shape == (2, 15)
    enc = evf.smor_to_float(sentences)
    assert enc.shape == (2, 14)
