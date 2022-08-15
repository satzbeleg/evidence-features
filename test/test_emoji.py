import evidence_features as evf


def test1():
    sentences = [
        "Dieser Satz ist ein Beispiel, aber eher kurz.",
        "Die Kuh macht muh, der Hund wufft aber lauter. 😀",
        "Psst 😍 ich habe einen neuen Account erstellt."
    ]
    enc = evf.emoji_to_int8(sentences)
    assert enc.shape == (3, 23)
    assert enc[1][-4:].sum() == 1
    assert enc[2][-4:].sum() == 1

    enc = evf.emoji_to_float(sentences)
    assert enc.shape == (3, 22)
    assert (enc[0] == 0.0).all()
