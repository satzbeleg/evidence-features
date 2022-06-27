import evidence_features as evf


def test1():
    sentences = [
        "Dieser Satz ist ein Beispiel, aber eher kurz.",
        "Die Kuh macht muh, der Hund wufft aber lauter."
    ]
    enc1, enc2, enc3 = evf.smor_to_int8(sentences)
    assert enc1.shape == (2, 7)
    assert enc2.shape == (2, 5)
    assert enc3.shape == (2, 5)
    enc1, enc2, enc3 = evf.smor_to_float(sentences)
    assert enc1.shape == (2, 6)
    assert enc2.shape == (2, 4)
    assert enc3.shape == (2, 4)
