import evidence_features as evf
import evidence_features.todisk

sentences = [
    "Dieser Satz ist ein Beispiel, aber eher kurz. Hier steht noch was",
    "Die Kuh macht muh, der Hund wufft aber lauter."
]

evf.todisk.encode_and_save(
    "test-check_todisk-sent.jsonl", sentences)

evf.todisk.encode_and_save(
    "test-check_todisk-doc.jsonl", sentences, document_level=True)
