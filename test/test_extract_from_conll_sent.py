import evidence_features as evf
import conllu


data = """
# sent_id = hdt-s23796
# text = Wer beim bevorstehenden Börsengang der Deutschen Post mit der " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich begehrten Papiere zu ergattern .
1	Wer	wer	PRON	PWS	Case=Nom|Number=Sing|PronType=Int,Rel	15	nsubj	_	_
2-3	beim	_	_	_	_	_	_	_	_
2	bei	bei	ADP	APPR	AdpType=Prep|Case=Dat	5	case	_	_
3	dem	der	DET	ART	Case=Dat|Definite=Def|Gender=Masc,Neut|Number=Sing|PronType=Art	5	det	_	_
4	bevorstehenden	bevorstehen	ADJ	ADJA	Degree=Pos|Number=Sing	5	amod	_	_
5	Börsengang	Gang	NOUN	NN	Gender=Masc|Number=Sing	15	obl	_	_
6	der	der	DET	ART	Case=Gen|Definite=Def|Gender=Fem|Number=Sing|PronType=Art	8	det	_	_
7	Deutschen	deutsch	ADJ	ADJA	Degree=Pos|Number=Sing	8	amod	_	_
8	Post	Post	NOUN	NN	Gender=Fem|Number=Sing	5	nmod	_	_
9	mit	mit	ADP	APPR	AdpType=Prep|Case=Dat	12	case	_	_
10	der	der	DET	ART	Case=Dat|Definite=Def|Gender=Fem|Number=Sing|PronType=Art	12	det	_	_
11	"	"	PUNCT	$(	PunctType=Brck	12	punct	_	_
12	Aktie	Aktie	NOUN	NN	Gender=Fem|Number=Sing	5	nmod	_	_
13	Gelb	Gelb	NOUN	NN	Gender=Neut|Number=Sing	12	appos	_	_
14	"	"	PUNCT	$(	PunctType=Brck	12	punct	_	_
15	liebäugelt	liebäugeln	VERB	VVFIN	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	17	csubj	_	_
16	,	,	PUNCT	$,	PunctType=Comm	15	punct	_	_
17	hat	haben	AUX	VAFIN	Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin	0	root	_	_
18	mit	mit	ADP	APPR	AdpType=Prep|Case=Dat	20	case	_	_
19	einem	ein	DET	ART	Case=Dat|Definite=Ind|Gender=Neut|Number=Sing|NumType=Card|PronType=Art	20	det	_	_
20	Depot	Depot	NOUN	NN	Gender=Neut|Number=Sing	17	obl	_	_
21	bei	bei	ADP	APPR	AdpType=Prep|Case=Dat	22	case	_	_
22	Postbank	Postbank	PROPN	NE	_	20	nmod	_	_
23	Easytrade	Easytrade	PROPN	NE	_	22	flat:name	_	_
24	bessere	gut	ADJ	ADJA	Degree=Cmp|Number=Plur	25	amod	_	_
25	Aussichten	Aussicht	NOUN	NN	Gender=Fem|Number=Plur	17	obj	_	_
26	,	,	PUNCT	$,	PunctType=Comm	33	punct	_	_
27	einige	einige	DET	PIS	Case=Acc|Number=Plur|PronType=Ind	33	obj	_	_
28	der	der	DET	ART	Case=Gen|Definite=Def|Number=Plur|PronType=Art	31	det	_	_
29	wahrscheinlich	wahrscheinlich	ADJ	ADJD	Degree=Pos|Variant=Short	30	advmod	_	_
30	begehrten	begehren	ADJ	ADJA	Degree=Pos|Number=Plur	31	amod	_	_
31	Papiere	Papier	NOUN	NN	Gender=Neut|Number=Plur	27	nmod	_	_
32	zu	zu	PART	PTKZU	PartType=Inf	33	mark	_	_
33	ergattern	ergattern	VERB	VVINF	VerbForm=Inf	25	xcomp	_	_
34	.	.	PUNCT	$.	PunctType=Peri	17	punct	_	_
"""


sentences = conllu.parse(data)


def test_words():
    sent = sentences[0]
    words = evf.get_words(sent)
    assert words == [
        'Wer',
        'beim',
        'bevorstehenden',
        'Börsengang',
        'der',
        'Deutschen',
        'Post',
        'mit',
        'der',
        'Aktie',
        'Gelb',
        'liebäugelt',
        'hat',
        'mit',
        'einem',
        'Depot',
        'bei',
        'Postbank',
        'Easytrade',
        'bessere',
        'Aussichten',
        'einige',
        'der',
        'wahrscheinlich',
        'begehrten',
        'Papiere',
        'zu',
        'ergattern']


def test_reconstructed():
    sent = sentences[0]
    reconstructed, lemmata, spans = evf.get_sentence_and_lemmata(
        sent, upos_list=["NOUN", "VERB", "ADJ", "DET"])
    assert reconstructed == 'Wer beim bevorstehenden Börsengang der Deutschen Post mit der " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich begehrten Papiere zu ergattern .'
    assert lemmata == [
        'bevorstehen',
        'Gang',
        'der',
        'deutsch',
        'Post',
        'der',
        'Aktie',
        'Gelb',
        'liebäugeln',
        'ein',
        'Depot',
        'gut',
        'Aussicht',
        'einige',
        'der',
        'wahrscheinlich',
        'begehren',
        'Papier',
        'ergattern']
    assert spans == [
        (9, 23),
        (24, 34),
        (35, 38),
        (39, 48),
        (49, 53),
        (58, 61),
        (64, 69),
        (70, 74),
        (77, 87),
        (98, 103),
        (104, 109),
        (133, 140),
        (141, 151),
        (154, 160),
        (161, 164),
        (165, 179),
        (180, 189),
        (190, 197),
        (201, 210)]


def test_feats12():
    sent = sentences[0]
    words = evf.get_words(sent)
    reconstructed, _, _ = evf.get_sentence_and_lemmata(sent)
    feats12 = evf.get_feats12(reconstructed, words)
    assert feats12 == (212, 28)


def test_group_spans():
    sent = sentences[0]

    _, lemmata0, spans0 = evf.get_sentence_and_lemmata(
        sent, upos_list=["NOUN", "VERB", "ADJ", "DET"])

    lemmata, spans = evf.group_lemma_spans(lemmata0, spans0)

    assert lemmata == [
        'Aktie',
        'Aussicht',
        'Depot',
        'Gang',
        'Gelb',
        'Papier',
        'Post',
        'begehren',
        'bevorstehen',
        'der',
        'deutsch',
        'ein',
        'einige',
        'ergattern',
        'gut',
        'liebäugeln',
        'wahrscheinlich']
    assert spans ==  [
        [(64, 69)],
        [(141, 151)],
        [(104, 109)],
        [(24, 34)],
        [(70, 74)],
        [(190, 197)],
        [(49, 53)],
        [(180, 189)],
        [(9, 23)],
        [(161, 164), (58, 61), (35, 38)],
        [(39, 48)],
        [(98, 103)],
        [(154, 160)],
        [(201, 210)],
        [(133, 140)],
        [(77, 87)],
        [(165, 179)]]
    

def test_masked():
    sent = sentences[0]

    reconstructed, lemmata0, spans0 = evf.get_sentence_and_lemmata(
        sent, upos_list=["NOUN", "VERB", "ADJ", "DET"])

    _, spans = evf.group_lemma_spans(lemmata0, spans0)

    masked = evf.get_masks(reconstructed, spans)

    assert masked == [
        'Wer beim bevorstehenden Börsengang der Deutschen Post mit der " [MASK] Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich begehrten Papiere zu ergattern .',
        'Wer beim bevorstehenden Börsengang der Deutschen Post mit der " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere [MASK] , einige der wahrscheinlich begehrten Papiere zu ergattern .',
        'Wer beim bevorstehenden Börsengang der Deutschen Post mit der " Aktie Gelb " liebäugelt , hat mit einem [MASK] bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich begehrten Papiere zu ergattern .',
        'Wer beim bevorstehenden [MASK] der Deutschen Post mit der " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich begehrten Papiere zu ergattern .',
        'Wer beim bevorstehenden Börsengang der Deutschen Post mit der " Aktie [MASK] " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich begehrten Papiere zu ergattern .',
        'Wer beim bevorstehenden Börsengang der Deutschen Post mit der " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich begehrten [MASK] zu ergattern .',
        'Wer beim bevorstehenden Börsengang der Deutschen [MASK] mit der " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich begehrten Papiere zu ergattern .',
        'Wer beim bevorstehenden Börsengang der Deutschen Post mit der " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich [MASK] Papiere zu ergattern .',
        'Wer beim [MASK] Börsengang der Deutschen Post mit der " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich begehrten Papiere zu ergattern .',
        'Wer beim bevorstehenden Börsengang [MASK] Deutschen Post mit [MASK] " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige [MASK] wahrscheinlich begehrten Papiere zu ergattern .',
        'Wer beim bevorstehenden Börsengang der [MASK] Post mit der " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich begehrten Papiere zu ergattern .',
        'Wer beim bevorstehenden Börsengang der Deutschen Post mit der " Aktie Gelb " liebäugelt , hat mit [MASK] Depot bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich begehrten Papiere zu ergattern .',
        'Wer beim bevorstehenden Börsengang der Deutschen Post mit der " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , [MASK] der wahrscheinlich begehrten Papiere zu ergattern .',
        'Wer beim bevorstehenden Börsengang der Deutschen Post mit der " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich begehrten Papiere zu [MASK] .',
        'Wer beim bevorstehenden Börsengang der Deutschen Post mit der " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade [MASK] Aussichten , einige der wahrscheinlich begehrten Papiere zu ergattern .',
        'Wer beim bevorstehenden Börsengang der Deutschen Post mit der " Aktie Gelb " [MASK] , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige der wahrscheinlich begehrten Papiere zu ergattern .',
        'Wer beim bevorstehenden Börsengang der Deutschen Post mit der " Aktie Gelb " liebäugelt , hat mit einem Depot bei Postbank Easytrade bessere Aussichten , einige der [MASK] begehrten Papiere zu ergattern .'
    ]


def test_feats2():
    sent = sentences[0]
    feats2 = evf.get_feats2(sent)
    assert feats2 == [35, 5, 4, 0, 1, 0, 6, 0, 7, 0, 1, 1, 2, 5, 0, 2, 1]


def test_feats3():
    sent = sentences[0]
    feats3 = evf.get_feats3(sent)
    assert feats3 == [35, 2, 2, 1, 0, 4, 0, 0, 0, 1, 5, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 1, 0, 2, 0, 0, 0, 5, 1, 4, 14, 6, 0, 0, 2, 1, 7, 2, 1, 4, 1, 0, 2, 0]


def test_adjac():
    sent = sentences[0]
    adjac = evf.get_adjac(sent)
    assert adjac == [
        (1, 15, 'nsubj'),
        (2, 5, 'case'),
        (3, 5, 'det'),
        (4, 5, 'amod'),
        (5, 15, 'obl'),
        (6, 8, 'det'),
        (7, 8, 'amod'),
        (8, 5, 'nmod'),
        (9, 12, 'case'),
        (10, 12, 'det'),
        (11, 12, 'punct'),
        (12, 5, 'nmod'),
        (13, 12, 'appos'),
        (14, 12, 'punct'),
        (15, 17, 'csubj'),
        (16, 15, 'punct'),
        (17, 0, 'root'),
        (18, 20, 'case'),
        (19, 20, 'det'),
        (20, 17, 'obl'),
        (21, 22, 'case'),
        (22, 20, 'nmod'),
        (23, 22, 'flat:name'),
        (24, 25, 'amod'),
        (25, 17, 'obj'),
        (26, 33, 'punct'),
        (27, 33, 'obj'),
        (28, 31, 'det'),
        (29, 30, 'advmod'),
        (30, 31, 'amod'),
        (31, 27, 'nmod'),
        (32, 33, 'mark'),
        (33, 25, 'xcomp'),
        (34, 17, 'punct')]
    

def test_edges():
    sent = sentences[0]
    edges = evf.get_edges(sent)
    assert edges == [
        (15, 1),
        (5, 2),
        (5, 3),
        (5, 4),
        (15, 5),
        (8, 6),
        (8, 7),
        (5, 8),
        (12, 9),
        (12, 10),
        (12, 11),
        (5, 12),
        (12, 13),
        (12, 14),
        (17, 15),
        (15, 16),
        (0, 17),
        (20, 18),
        (20, 19),
        (17, 20),
        (22, 21),
        (20, 22),
        (22, 23),
        (25, 24),
        (17, 25),
        (33, 26),
        (33, 27),
        (31, 28),
        (30, 29),
        (31, 30),
        (27, 31),
        (33, 32),
        (25, 33),
        (17, 34)]
