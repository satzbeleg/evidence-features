import evidence_features as evf


def test1():
    sentences = [
        "Dieser Satz ist ein Beispiel, aber eher kurz.",
        "Die Kuh macht muh, der Hund wufft aber lauter."
    ]
    hsh = evf.kshingle_to_int32(sentences)
    assert hsh.shape == (2, 32)
