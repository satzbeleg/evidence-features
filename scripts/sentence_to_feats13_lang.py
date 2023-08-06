import fasttext
import ipasymbols
from typing import List
import logging
from timeit import default_timer as timer
import argparse
import ray
import gc
import os
import psutil
import jsonlines



logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(name)s: %(message)s",
    datefmt="%y-%m-%d %H:%M:%S"
)


# parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--input-file', default="./extracted.jsonl", type=str
)
parser.add_argument(
    '--output-file', default="./feats13.jsonl", type=str
)
parser.add_argument(
    '--batch-size', default=500000, type=int
)
parser.add_argument(
    '--model-folder', default="./models", type=str
)
args = parser.parse_args()


# CPU settings
PCT_CPU = float(os.environ.get("LANG_PCT_CPU", "0.9"))
NUM_CPU = os.environ.get("LANG_NUM_CPU")
if NUM_CPU is None:
    NUM_CPU = max(1, int(psutil.cpu_count(logical=False) * PCT_CPU))
ray.init(num_cpus=NUM_CPU)

# path to the SMOR datasets
MODELPATH = os.path.join(args.model_folder, "lid.176.ftz")

# Load the pretrained model
model_ft176 = fasttext.FastText._FastText(model_path=MODELPATH)


# util fun
LANGS = [
    ["de"], ["nds"], ["als"], ["bar"],
    ["nl", "zea", "vls", "li", "af"] + ["lb"],
    ["da", "sv", "no", "nn", "is"],
    ["en", "sco", "frr", "fy"],
    ["wa", "fr", "rm", "it"],
    ["hsb", "pl", "cs", "sk", "sl"]
]


def lang_to_id(lang: str) -> int:
    # lang_to_id("wa"), lang_to_id("xa")
    for i, GRP in enumerate(LANGS):
        if lang in GRP:
            return i
    return len(LANGS)


def lookup_lang(labels: List[str], probas: List[float]):
    # extract the tags
    labels = [s.split("__")[-1] for s in labels]
    # init output
    out = [0.0 for _ in range(len(LANGS) + 1)]
    # loop over results
    for i, lang in enumerate(labels):
        grpid = lang_to_id(lang)
        out[grpid] = max(out[grpid], probas[i])
    # done
    return out


BRACKETS = [
    0.00390625, 0.0078125, 0.01171875, 0.015625, 0.01953125, 0.0234375, 
    0.02734375, 0.03125, 0.03515625, 0.0390625, 0.04296875, 0.046875, 
    0.05078125, 0.0546875, 0.05859375, 0.0625, 0.06640625, 0.0703125, 
    0.07421875, 0.078125, 0.08203125, 0.0859375, 0.08984375, 0.09375, 
    0.09765625, 0.1015625, 0.10546875, 0.109375, 0.11328125, 0.1171875, 
    0.12109375, 0.125, 0.12890625, 0.1328125, 0.13671875, 0.140625, 
    0.14453125, 0.1484375, 0.15234375, 0.15625, 0.16015625, 0.1640625, 
    0.16796875, 0.171875, 0.17578125, 0.1796875, 0.18359375, 0.1875, 
    0.19140625, 0.1953125, 0.19921875, 0.203125, 0.20703125, 0.2109375, 
    0.21484375, 0.21875, 0.22265625, 0.2265625, 0.23046875, 0.234375, 
    0.23828125, 0.2421875, 0.24609375, 0.25, 0.25390625, 0.2578125, 
    0.26171875, 0.265625, 0.26953125, 0.2734375, 0.27734375, 0.28125, 
    0.28515625, 0.2890625, 0.29296875, 0.296875, 0.30078125, 0.3046875, 
    0.30859375, 0.3125, 0.31640625, 0.3203125, 0.32421875, 0.328125, 
    0.33203125, 0.3359375, 0.33984375, 0.34375, 0.34765625, 0.3515625, 
    0.35546875, 0.359375, 0.36328125, 0.3671875, 0.37109375, 0.375, 
    0.37890625, 0.3828125, 0.38671875, 0.390625, 0.39453125, 0.3984375, 
    0.40234375, 0.40625, 0.41015625, 0.4140625, 0.41796875, 0.421875, 
    0.42578125, 0.4296875, 0.43359375, 0.4375, 0.44140625, 0.4453125, 
    0.44921875, 0.453125, 0.45703125, 0.4609375, 0.46484375, 0.46875, 
    0.47265625, 0.4765625, 0.48046875, 0.484375, 0.48828125, 0.4921875, 
    0.49609375, 0.5, 0.50390625, 0.5078125, 0.51171875, 0.515625, 0.51953125, 
    0.5234375, 0.52734375, 0.53125, 0.53515625, 0.5390625, 0.54296875, 
    0.546875, 0.55078125, 0.5546875, 0.55859375, 0.5625, 0.56640625, 
    0.5703125, 0.57421875, 0.578125, 0.58203125, 0.5859375, 0.58984375, 
    0.59375, 0.59765625, 0.6015625, 0.60546875, 0.609375, 0.61328125, 
    0.6171875, 0.62109375, 0.625, 0.62890625, 0.6328125, 0.63671875, 
    0.640625, 0.64453125, 0.6484375, 0.65234375, 0.65625, 0.66015625, 
    0.6640625, 0.66796875, 0.671875, 0.67578125, 0.6796875, 0.68359375, 
    0.6875, 0.69140625, 0.6953125, 0.69921875, 0.703125, 0.70703125, 
    0.7109375, 0.71484375, 0.71875, 0.72265625, 0.7265625, 0.73046875, 
    0.734375, 0.73828125, 0.7421875, 0.74609375, 0.75, 0.75390625, 
    0.7578125, 0.76171875, 0.765625, 0.76953125, 0.7734375, 0.77734375, 
    0.78125, 0.78515625, 0.7890625, 0.79296875, 0.796875, 0.80078125, 
    0.8046875, 0.80859375, 0.8125, 0.81640625, 0.8203125, 0.82421875, 
    0.828125, 0.83203125, 0.8359375, 0.83984375, 0.84375, 0.84765625, 
    0.8515625, 0.85546875, 0.859375, 0.86328125, 0.8671875, 0.87109375, 
    0.875, 0.87890625, 0.8828125, 0.88671875, 0.890625, 0.89453125, 
    0.8984375, 0.90234375, 0.90625, 0.91015625, 0.9140625, 0.91796875, 
    0.921875, 0.92578125, 0.9296875, 0.93359375, 0.9375, 0.94140625,
    0.9453125, 0.94921875, 0.953125, 0.95703125, 0.9609375, 0.96484375, 
    0.96875, 0.97265625, 0.9765625, 0.98046875, 0.984375, 0.98828125, 
    0.9921875, 0.99609375, 1.0]


def find_percentile(f, brackets):
    for i, b in enumerate(brackets):
        if f <= b:
            return i
    return len(brackets) - 1

def scaledfloat_to_int8(x: float) -> int:
    """ Convert a scaled [0.0, 1.0] float to an 8-bit integer.
    Example:
        idx = scaledfloat_to_int8(1.0)
        idx, int8_to_scaledfloat(idx)
    """
    x = min(1.0, max(0.0, x))
    return find_percentile(x, BRACKETS) - 128


@ray.remote
def postprocess_to_int8_wrapper(lb, pb):
    # assign labels/langs to language groups
    pdf = lookup_lang(lb, pb)
    # encode to int8
    encoded = [scaledfloat_to_int8(p) for p in pdf]
    return encoded


def postprocess_to_int8(labels, probas):
    try:
        return ray.get([
            postprocess_to_int8_wrapper.remote(lb, pb.tolist())
            for lb, pb in zip(*(labels, probas))])
    except Exception as e:
        logger.error(e)
        ray.shutdown()
        gc.collect()


if __name__ == '__main__':
    logger.info("Start")
    start = timer()

    # read data
    with jsonlines.open(args.input_file) as reader:
        batch_sentences, data = [], []
        for obj in reader:
            sent = obj.pop("sentence")
            sent_id = obj.pop("sent_id")
            batch_sentences.append(sent)
            data.append({"sent_id": sent_id})

            # start processing the batch
            if len(data) == args.batch_size:
                # process with fasttext
                labels, probas = model_ft176.predict(batch_sentences, k=3)
                del batch_sentences
                # postprocess with ray.io
                results = postprocess_to_int8(labels, probas)
                del labels, probas

                # write data
                with jsonlines.open(args.output_file, mode='a') as writer:
                    for obj, res in zip(data, results):
                        obj["feats13"] = res
                        writer.write(obj)

                del results, data
                batch_sentences, data = [], []

    if len(data) > 0:
        labels, probas = model_ft176.predict(batch_sentences, k=3)
        results = postprocess_to_int8(labels, probas)
        with jsonlines.open(args.output_file, mode='a') as writer:
            for obj, res in zip(data, results):
                obj["feats13"] = res
                writer.write(obj)

    # done
    ray.shutdown()
    gc.collect()
    logger.info(f"End: {timer() - start: .6f} sec. elapsed")

