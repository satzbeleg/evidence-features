import conllu
from typing import List
import io
from .extract_from_conll_sent import extract_from_sentence
import os
import jsonlines
import logging

# start logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    filename="layerwise-training-v0.6.2.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    datefmt="%y-%m-%d %H:%M:%S"
)


def get_ddc_license(sent: conllu.TokenList) -> str:
	return sent.metadata.get('DDC:meta.availability')


def get_ddc_biblio(sent: conllu.TokenList):
	if sent.metadata.get('DDC:meta.textClass', "").lower() == "zeitung":
		dat = [
			sent.metadata.get('DDC:meta.author'),
			sent.metadata.get('DDC:meta.title'),
			sent.metadata.get('DDC:meta.bibl')
		]
		dat = [d for d in dat if d]
		biblio = ". ".join(dat)
	else:
		biblio = sent.metadata.get('DDC:meta.bibl')
	return biblio


def write_to_disk(OUTPUTFOLDER: str,
		  		  batch_biblio: List[dict],
		          batch_extracted: List[dict],
		          batch_words: List[dict],
		          batch_masked: List[dict],
		          batch_adjac: List[dict],
		          batch_edges: List[dict]):
	try:
		# biblio
		if len(batch_biblio) > 0:
			FILE1 = os.path.join(OUTPUTFOLDER, "biblio.jsonl")
			with jsonlines.open(FILE1, mode='a') as writer1:
				writer1.write_all(batch_biblio)
			batch_biblio = []
		# extracted
		if len(batch_extracted) > 0:
			FILE2 = os.path.join(OUTPUTFOLDER, "extracted.jsonl")
			with jsonlines.open(FILE2, mode='a') as writer2:
				writer2.write_all(batch_extracted)
			batch_extracted = []
		# words
		if len(batch_words) > 0:
			FILE3 = os.path.join(OUTPUTFOLDER, "words.jsonl")
			with jsonlines.open(FILE3, mode='a') as writer3:
				writer3.write_all(batch_words)
			batch_words = []
		# masked
		if len(batch_masked) > 0:
			FILE4 = os.path.join(OUTPUTFOLDER, "masked.jsonl")
			with jsonlines.open(FILE4, mode='a') as writer4:
				writer4.write_all(batch_masked)
			batch_masked = []
		# adjac
		if len(batch_adjac) > 0:
			FILE5 = os.path.join(OUTPUTFOLDER, "adjac.jsonl")
			with jsonlines.open(FILE5, mode='a') as writer5:
				writer5.write_all(batch_adjac)
			batch_adjac = []
		# edges
		if len(batch_edges) > 0:
			FILE6 = os.path.join(OUTPUTFOLDER, "edges.jsonl")
			with jsonlines.open(FILE6, mode='a') as writer6:
				writer6.write_all(batch_edges)
			batch_edges = []
	except Exception as e:
		logger.error(f"{len(batch_extracted)} examples not processed")
		logger.error(e)
	# reset batch
	batch_biblio = []
	batch_extracted = []
	batch_words = []
	batch_masked = []
	batch_adjac = []
	batch_edges = []
	# done
	return (
		batch_biblio, batch_extracted, batch_words, 
		batch_masked, batch_adjac, batch_edges)


def read_conll_file(CONLLFILE: str, 
		    		OUTPUTFOLDER: str,
		            IO_BATCH_SIZE: int=1000000):
	# open file pointer to conll file
	with io.open(CONLLFILE, "r", encoding="utf-8") as fp:
		# start settings
		license, biblio = None, None

		# tmp batches
		batch_biblio = []
		batch_extracted = []
		batch_words = []
		batch_masked = []
		batch_adjac = []
		batch_edges = []

		# loop over each sentence
		for sent in conllu.parse_incr(fp):
			# replace new license info
			license_tmp = get_ddc_license(sent)
			if license_tmp is not None:
				if license is None:  # 1st occurence
					license = license_tmp
				else:
					if license_tmp != license:
						license = license_tmp

			# replace new biblio info
			biblio_tmp = get_ddc_biblio(sent)
			if biblio_tmp is not None:
				if biblio is None:  # 1st occurence
					biblio = biblio_tmp
					batch_biblio.append(biblio)
				else:
					if biblio_tmp != biblio:  # new biblio
						biblio = biblio_tmp
						batch_biblio.append(biblio)

            # extract data from conll sentence
			extracted, words, masked, adjac, edges = extract_from_sentence(sent)
			# add license
			extracted['license'] = license if license else "no license"
			extracted['biblio'] = biblio if biblio else "unknown source"
			# append to batch
			batch_extracted.append(extracted)
			batch_words.append(words)
			batch_masked.extend(masked)  # it's a list!
			batch_adjac.append(adjac)
			batch_edges.append(edges)

			# Write batches to disk
			if len(batch_extracted) >= IO_BATCH_SIZE:
				# write to disk inkl. list reset
				(
					batch_biblio, batch_extracted, batch_words, 
					batch_masked, batch_adjac, batch_edges
				) = write_to_disk(
					OUTPUTFOLDER, 
					batch_biblio, batch_extracted, batch_words, 
					batch_masked, batch_adjac, batch_edges)

		# save last batch
		if len(batch_extracted) > 0:
			write_to_disk(
				OUTPUTFOLDER, 
				batch_biblio, batch_extracted, batch_words, 
				batch_masked, batch_adjac, batch_edges)
