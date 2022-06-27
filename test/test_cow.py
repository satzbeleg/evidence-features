import evidence_features as evf


def test1():
    sentences = [
        "Dieser Satz ist ein Beispiel, aber eher kurz.",
        "Die Kuh macht muh, der Hund wufft aber lauter."
    ]
    encoded = evf.derechar_to_int16(sentences)
    assert encoded.shape == (2, 7)
    encoded = evf.derechar_to_float(sentences)
    assert encoded.shape == (2, 6)

    encoded = evf.derebigram_to_int16(sentences)
    assert encoded.shape == (2, 11)
    encoded = evf.derebigram_to_float(sentences)
    assert encoded.shape == (2, 10)
