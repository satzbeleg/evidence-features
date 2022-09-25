import evidence_features as evf


def test1():
    sentences = [
        "Dieser Satz ist ein Beispiel, aber eher kurz.",
        "Die Kuh macht muh, der Hund wufft aber lauter."
    ]
    enc1, enc2, enc3, hsh4, lem5 = evf.trankit_to_int(
        sentences, skiphash=False)
    assert enc1.shape == (2, 17)
    assert enc2.shape == (2, 48)
    assert enc3.shape == (2, 21)
    assert hsh4.shape == (2, 32)
    enc1, enc2, enc3 = evf.trankit_to_int(sentences)
    assert enc1.shape == (2, 17)
    assert enc2.shape == (2, 48)
    assert enc3.shape == (2, 21)
    enc1, enc2, enc3 = evf.trankit_to_float(sentences)
    assert enc1.shape == (2, 16)
    assert enc2.shape == (2, 47)
    assert enc3.shape == (2, 21)
