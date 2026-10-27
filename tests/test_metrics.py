import copy

from care_record.metrics import safety_error_score, score_case


def test_perfect_prediction(good_note):
    score = score_case(good_note, good_note, [], set())
    crit = score.critical()
    assert crit.correct == crit.n_truth == 4
    assert crit.precision == crit.recall == crit.f1 == 1.0
    assert safety_error_score(crit) == 0


def test_hallucination_wrong_value_and_miss(good_note):
    pred = copy.deepcopy(good_note)
    pred["vital_signs"][1]["value"] = 76  # wrong value
    pred["medications"].append({"name": "Metformin", "status": "given", "evidence": "x"})  # hallucination
    pred["incidents"] = []  # miss
    score = score_case(pred, good_note, [], {"medications[1]"})
    crit = score.critical()
    assert (crit.correct, crit.wrong_value, crit.hallucinated, crit.missed) == (2, 1, 1, 1)
    assert score.hallucinated_paths == ["medications[1]"]
    assert score.hallucinations_flagged == 1
    assert safety_error_score(crit) == (3 * 1 + 2 * 1 + 1 * 1) / 4


def test_expect_review_items_match_on_entity_only(good_note):
    pred = copy.deepcopy(good_note)
    pred["vital_signs"][1]["value"] = 76
    score = score_case(pred, good_note, ["vital_signs[1]"], {"vital_signs[1]"})
    assert score.critical().wrong_value == 0
    assert score.expected_review_flagged == 1


def test_medication_aliases_match(good_note):
    truth = copy.deepcopy(good_note)
    truth["medications"][0]["name"] = "Novaminsulfon"
    pred = copy.deepcopy(truth)
    pred["medications"][0]["name"] = "Novalgin"
    assert score_case(pred, truth, [], set()).sections["medications"].correct == 1
