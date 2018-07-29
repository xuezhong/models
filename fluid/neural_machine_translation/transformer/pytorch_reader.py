import glob
import torch 
from onmt.utils.logging import logger

from onmt.inputters.inputter  import load_fields_from_vocab, DatasetLazyIter

class PytorchReaderConfig(object):
    data = 'data/demo1'
    batch_size = 3
    valid_batch_size = 128
    batch_type = "sents"
    gpuid = 1

def my_build_dataset_iter(datasets, fields, config, is_train=True):
    """
    This returns user-defined train/validate data iterator for the trainer
    to iterate over. We implement simple ordered iterator strategy here,
    but more sophisticated strategy like curriculum learning is ok too.
    """
    batch_size = config.batch_size if is_train else config.valid_batch_size
    if is_train and config.batch_type == "tokens":
        def batch_size_fn(new, count, sofar):
            """
            In token batching scheme, the number of sequences is limited
            such that the total number of src/tgt tokens (including padding)
            in a batch <= batch_size
            """
            # Maintains the longest src and tgt length in the current batch
            global max_src_in_batch, max_tgt_in_batch
            # Reset current longest length at a new batch (count=1)
            if count == 1:
                max_src_in_batch = 0
                max_tgt_in_batch = 0
            # Src: <bos> w1 ... wN <eos>
            max_src_in_batch = max(max_src_in_batch, len(new.src) + 2)
            # Tgt: w1 ... wN <eos>
            max_tgt_in_batch = max(max_tgt_in_batch, len(new.tgt) + 1)
            src_elements = count * max_src_in_batch
            tgt_elements = count * max_tgt_in_batch
            return max(src_elements, tgt_elements)
    else:
        batch_size_fn = None
    # device = config.device_id if config.gpuid else -1
    # breaking change torchtext 0.3
    if config.gpuid:
        device = "cuda"
    else:
        device = "cpu"

    return DatasetLazyIter(datasets, fields, batch_size, batch_size_fn,
                           device, is_train)


def my_lazily_load_dataset(corpus_type, config):
    """
    Dataset generator. Don't do extra stuff here, like printing,
    because they will be postponed to the first loading time.

    Args:
        corpus_type: 'train' or 'valid'
    Returns:
        A list of dataset, the dataset(s) are lazily loaded.
    """
    assert corpus_type in ["train", "valid"]

    def _lazy_dataset_loader(pt_file, corpus_type):
        dataset = torch.load(pt_file)
        logger.info('Loading %s dataset from %s, number of examples: %d' %
                    (corpus_type, pt_file, len(dataset)))
        return dataset

    # Sort the glob output by file name (by increasing indexes).
    pts = sorted(glob.glob(config.data + '.' + corpus_type + '.[0-9]*.pt'))
    if pts:
        for pt in pts:
            yield _lazy_dataset_loader(pt, corpus_type)
    else:
        # Only one inputters.*Dataset, simple!
        pt = config.data + '.' + corpus_type + '.pt'
        yield _lazy_dataset_loader(pt, corpus_type)

def my_load_fields(dataset, data_type, config, checkpoint):
    if checkpoint is not None:
        logger.info('Loading vocab from checkpoint at %s.' % config.train_from)
        fields = load_fields_from_vocab(
            checkpoint['vocab'], data_type)
    else:
        fields = load_fields_from_vocab(
            torch.load(config.data + '.vocab.pt'), data_type)
    fields = dict([(k, f) for (k, f) in fields.items()
                   if k in dataset.examples[0].__dict__])

    if data_type == 'text':
        logger.info(' * vocabulary size. source = %d; target = %d' %
                    (len(fields['src'].vocab), len(fields['tgt'].vocab)))
    else:
        logger.info(' * vocabulary size. target = %d' %
                    (len(fields['tgt'].vocab)))

    return fields

def pytorch_reader(config):
    checkpoint = None
    # Peek the fisrt dataset to determine the data_type.
    # (All datasets have the same data_type).
    first_dataset = next(my_lazily_load_dataset("train", config))
    data_type = first_dataset.data_type

    # Load fields generated from preprocess phase.
    fields = my_load_fields(first_dataset, data_type, config, checkpoint)


    def my_train_iter_fct(): return my_build_dataset_iter(
        my_lazily_load_dataset("train", config), fields, config)

    def my_valid_iter_fct(): return my_build_dataset_iter(
        my_lazily_load_dataset("valid", config), fields, config)

    return my_train_iter_fct, fields
