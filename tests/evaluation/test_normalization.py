from neoolaf.evaluation.matching.normalization import normalize_text


def test_normalize_text_splits_camel_case():
    assert normalize_text("OpenSideGuardAlarmEvent") == "open side guard alarm event"


def test_normalize_text_removes_punctuation():
    assert normalize_text("THERMOMAGNET-SWITCHES. END") == "thermomagnet switches end"
