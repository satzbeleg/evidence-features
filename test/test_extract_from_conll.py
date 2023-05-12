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


def test1():
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


