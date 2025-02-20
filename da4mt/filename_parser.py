import unittest
from pathlib import Path
from typing import NamedTuple, Optional


class ModelComponents(NamedTuple):
    dataset: Optional[str]
    domain_adaptation: Optional[str]
    pretraining: str
    train_size: int


def parse_model_name(file_path: Path) -> ModelComponents:
    """
    Parse a model name from a file path and extract its components.

    This function takes a file path, extracts the model name, and parses it
    into its constituent parts: dataset (optional), domain adaptation (optional),
    pretraining, and train size. The dataset is only optional if domain adaptation
    is also optional.

    :param file_path: A Path object representing the file path to parse.
    :return: A ModelComponents named tuple containing the parsed components of the model name.

    :raises ValueError: If the input string doesn't match the expected format.

    Examples:
    >>> from pathlib import Path
    >>> parse_model_name(Path("adme_solubility_cbert_none-bert-0"))
    ModelComponents(dataset='adme_solubility', domain_adaptation='cbert', pretraining='none', train_size=0)
    >>> parse_model_name(Path("none-bert-30"))
    ModelComponents(dataset=None, domain_adaptation=None, pretraining='none', train_size=30)
    >>> parse_model_name(Path("complex_dataset_name_with_underscores_cbert_none-bert-50"))
    ModelComponents(dataset='complex_dataset_name_with_underscores', domain_adaptation='cbert', pretraining='none', train_size=50)
    """
    # Extract the file name without extension
    model_name = file_path.stem

    # Split the model name into components
    parts = model_name.split("-bert-")
    if len(parts) != 2:
        raise ValueError("Invalid model name format")

    main_part, train_size = parts

    # Split the main part
    main_components = main_part.split("_")

    # Optional components
    dataset = None
    domain_adaptation = None

    # Determine the number of components and assign accordingly
    if len(main_components) == 1:
        pretraining = main_components[0]
    elif len(main_components) >= 3:
        pretraining = main_components[-1]
        domain_adaptation = main_components[-2]
        dataset = "_".join(main_components[:-2])
    else:
        raise ValueError("Invalid model name format")

    # Validate components
    if domain_adaptation and domain_adaptation not in {"cbert", "mlm", "sbert", "mtr"}:
        raise ValueError("Invalid domain adaptation")
    if pretraining not in {"none", "mlm", "mtr"}:
        raise ValueError("Invalid pretraining")

    try:
        train_size_int = int(train_size)
        if train_size_int < 0:
            raise ValueError("Train size must be non-negative")
    except ValueError:
        raise ValueError("Invalid train size")

    return ModelComponents(
        dataset=dataset,
        domain_adaptation=domain_adaptation,
        pretraining=pretraining,
        train_size=train_size_int,
    )


class TestParseModelName(unittest.TestCase):
    def test_valid_inputs(self):
        test_cases = [
            (
                "adme_solubility_cbert_none-bert-0",
                ModelComponents(
                    dataset="adme_solubility",
                    domain_adaptation="cbert",
                    pretraining="none",
                    train_size=0,
                ),
            ),
            (
                "bace_mlm_none-bert-0",
                ModelComponents(
                    dataset="bace",
                    domain_adaptation="mlm",
                    pretraining="none",
                    train_size=0,
                ),
            ),
            (
                "bbbp_sbert_mlm-bert-30",
                ModelComponents(
                    dataset="bbbp",
                    domain_adaptation="sbert",
                    pretraining="mlm",
                    train_size=30,
                ),
            ),
            (
                "adme_ppb_h_cbert_mtr-bert-100",
                ModelComponents(
                    dataset="adme_ppb_h",
                    domain_adaptation="cbert",
                    pretraining="mtr",
                    train_size=100,
                ),
            ),
            (
                "none-bert-30",
                ModelComponents(
                    dataset=None,
                    domain_adaptation=None,
                    pretraining="none",
                    train_size=30,
                ),
            ),
            (
                "mlm-bert-100",
                ModelComponents(
                    dataset=None,
                    domain_adaptation=None,
                    pretraining="mlm",
                    train_size=100,
                ),
            ),
            (
                "complex_dataset_name_with_underscores_cbert_none-bert-50",
                ModelComponents(
                    dataset="complex_dataset_name_with_underscores",
                    domain_adaptation="cbert",
                    pretraining="none",
                    train_size=50,
                ),
            ),
        ]

        for file_name, expected_output in test_cases:
            with self.subTest(file_name=file_name):
                result = parse_model_name(Path(file_name))
                self.assertEqual(result, expected_output)

    def test_invalid_format(self):
        invalid_names = [
            "invalid_name",
            "missing_bert_separator_0",
            "extra_bert_cbert_none-bert-bert-0",
            "-bert-0",  # Empty pretraining
            "cbert_mlm-bert-0",  # Only domain adaptation without dataset
            "dataset_cbert-bert-0",  # Missing pretraining
        ]

        for name in invalid_names:
            with self.subTest(name=name):
                with self.assertRaises(
                    ValueError, msg=f"Should raise ValueError for {name}"
                ):
                    parse_model_name(Path(name))

    def test_invalid_domain_adaptation(self):
        with self.assertRaises(ValueError):
            parse_model_name(Path("dataset_invalid_none-bert-0"))

    def test_invalid_pretraining(self):
        with self.assertRaises(ValueError):
            parse_model_name(Path("dataset_cbert_invalid-bert-0"))

    def test_invalid_train_size(self):
        invalid_sizes = [
            "dataset_cbert_none-bert-invalid",
            "dataset_cbert_none-bert--1",
        ]
        for name in invalid_sizes:
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    parse_model_name(Path(name))

    def test_complex_dataset_name(self):
        result = parse_model_name(
            Path("complex_dataset_name_with_underscores_cbert_none-bert-50")
        )
        self.assertEqual(result.dataset, "complex_dataset_name_with_underscores")
        self.assertEqual(result.domain_adaptation, "cbert")
        self.assertEqual(result.pretraining, "none")
        self.assertEqual(result.train_size, 50)

    def test_large_train_size(self):
        result = parse_model_name(Path("dataset_cbert_none-bert-1000000"))
        self.assertEqual(result.train_size, 1000000)

    def test_namedtuple_attributes(self):
        result = parse_model_name(Path("adme_solubility_cbert_none-bert-0"))
        self.assertTrue(hasattr(result, "dataset"))
        self.assertTrue(hasattr(result, "domain_adaptation"))
        self.assertTrue(hasattr(result, "pretraining"))
        self.assertTrue(hasattr(result, "train_size"))

    def test_optional_components(self):
        result = parse_model_name(Path("none-bert-0"))
        self.assertIsNone(result.dataset)
        self.assertIsNone(result.domain_adaptation)
        self.assertEqual(result.pretraining, "none")
        self.assertEqual(result.train_size, 0)


if __name__ == "__main__":
    unittest.main()
