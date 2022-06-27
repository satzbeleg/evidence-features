import evidence_features as evf


def test1():
    sentences = [
        "Dieser Satz ist ein Beispiel, aber eher kurz.",
        "Die Kuh macht muh, der Hund wufft aber lauter."
    ]
    encoded = evf.cow_to_int8(sentences)
    assert encoded.shape == (2, 7)
    encoded = evf.cow_to_float(sentences)
    assert encoded.shape == (2, 6)
