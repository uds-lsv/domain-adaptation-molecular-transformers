"""BERT WordPiece tokenizer training for SMILES strings."""
import logging

from tokenizers.implementations import BertWordPieceTokenizer


def train_tokenizer(dataset_path: str, output_path: str, name, logger: logging.Logger):
    """Train a BERT WordPiece tokenizer on a SMILES dataset.

    :param str dataset_path: Path to the training file (one SMILES per line).
    :param str output_path: Directory where the vocabulary file will be saved.
    :param name: Name prefix for the output vocabulary file.
    :param logging.Logger logger: Logger instance.
    """
    tokenizer = BertWordPieceTokenizer(
        clean_text=True,
        handle_chinese_chars=False,
        strip_accents=False,
        lowercase=False,
    )
    logger.info(f"Training tokenizer on {dataset_path}")
    tokenizer.train(
        files=dataset_path,
        vocab_size=4096,
        min_frequency=2,
        limit_alphabet=1000,
        wordpieces_prefix="##",
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )

    logger.info(f"Saving tokenizer to {output_path}/{name}-vocab.txt")
    tokenizer.save_model(output_path, name)
