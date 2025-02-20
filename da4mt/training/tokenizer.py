import logging

from tokenizers.implementations import BertWordPieceTokenizer


def train_tokenizer(dataset_path: str, output_path: str, name, logger: logging.Logger):
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
