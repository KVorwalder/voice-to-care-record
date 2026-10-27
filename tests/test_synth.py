from care_record.numbers_de import de_words, numbers_in
from care_record.schema import ExtractedNote
from care_record.synth import CATEGORY_COUNTS, generate_dataset


def test_de_words():
    assert de_words(21) == "einundzwanzig"
    assert de_words(130) == "hundertdreißig"
    assert de_words(245) == "zweihundertfünfundvierzig"


def test_numbers_in_digits_and_words():
    found = numbers_in("Temperatur achtunddreißig komma fünf, RR hundertzwanzig zu 80")
    assert 38.5 in found and 120 in found and 80 in found


def test_dataset_is_deterministic():
    assert generate_dataset(1) == generate_dataset(1)
    assert generate_dataset(1) != generate_dataset(2)


def test_dataset_truth_is_valid_and_grounded():
    cases = generate_dataset()
    assert len(cases) == sum(CATEGORY_COUNTS.values())
    for case in cases:
        note = ExtractedNote.model_validate(case["truth"])
        for section in ("observations", "vital_signs", "medications", "incidents"):
            for item in getattr(note, section):
                assert item.evidence in case["transcript"]


def test_nothing_noteworthy_has_empty_truth():
    for case in generate_dataset():
        if case["category"] == "nothing_noteworthy":
            assert not any(case["truth"][s] for s in ("observations", "vital_signs", "medications", "incidents"))
