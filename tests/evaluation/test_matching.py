from neoolaf.evaluation.matching.entity_matching import greedy_entity_matching
from neoolaf.evaluation.profiles.registry import get_profile


def test_greedy_entity_matching_basic():
    profile = get_profile("general_relation_extraction")
    result = greedy_entity_matching({"side guard open"}, {"Side Guard Open"}, profile)
    assert result.prf.tp == 1
    assert result.prf.f1 == 1.0
