import evidence_features as evf


def test1():
    sentences = [
        "Dieser Satz ist ein Beispiel, aber eher kurz.",
        "Die Kuh macht muh, der Hund wufft aber lauter."
    ]
    encoded = evf.consonant_to_int16(sentences)
    assert encoded.shape == (2, 4)
    encoded = evf.consonant_to_float(sentences)
    assert encoded.shape == (2, 3)
