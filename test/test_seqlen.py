import evidence_features as evf


def test1():
    sentences = [
        "Dieser Satz ist ein Beispiel, aber eher kurz.",
        "Die Kuh macht muh, der Hund wufft aber lauter."
    ]
    encoded = evf.seqlen_to_int16(sentences)
    assert encoded.shape == (2, 2)
    encoded = evf.seqlen_to_float(sentences)
    assert encoded.shape == (2, 2)
