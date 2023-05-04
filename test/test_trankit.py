import evidence_features as evf


def test1():
    sentences = [
        "Dieser Satz ist ein Beispiel, aber eher kurz. Und ein extra Satzbeleg",
        "Die Kuh macht muh, der Hund wufft aber lauter."
    ]

    (
        enc1, enc2, enc3, hsh4,
        sents, lem5, masks6, span7, ann8
    ) = evf.trankit_to_int(sentences)

    assert enc1.shape == (2, 17)
    assert enc2.shape == (2, 48)
    assert enc3.shape == (2, 21)
    assert hsh4.shape == (2, 32)
    assert len(lem5) == 2
    assert len(sents) == 2
    assert len(masks6) == 2
    assert len(span7) == 2
    assert len(ann8) == 2

    enc1, enc2, enc3 = evf.trankit_to_float(sentences)
    assert enc1.shape == (2, 16)
    assert enc2.shape == (2, 47)
    assert enc3.shape == (2, 21)


def test2():
    sentences = [
        "Dieser Satz ist ein Beispiel, aber eher kurz. Und ein extra Satzbeleg",
        "Die Kuh macht muh, der Hund wufft aber lauter."
    ]

    (
        enc1, enc2, enc3, hsh4,
        sents, lem5, masks6, span7, ann8
    ) = evf.trankit_to_int(sentences, document_level=True)

    assert enc1.shape == (3, 17)
    assert enc2.shape == (3, 48)
    assert enc3.shape == (3, 21)
    assert hsh4.shape == (3, 32)
    assert len(lem5) == 3
    assert len(sents) == 3
    assert len(masks6) == 3
    assert len(span7) == 3
    assert len(ann8) == 3

    enc1, enc2, enc3 = evf.trankit_to_float(sentences, document_level=True)
    assert enc1.shape == (3, 16)
    assert enc2.shape == (3, 47)
    assert enc3.shape == (3, 21)
