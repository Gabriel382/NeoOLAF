from __future__ import annotations

# Local imports
from neoolaf.domain.user_guidance import UserGuidance


def get_effective_promotion_threshold(guidance: UserGuidance | None) -> float:
    """
    Return the effective promotion threshold based on ontology depth and user bias.
    """
    if guidance is None:
        return 0.50

    base = guidance.promotion_min_confidence

    if guidance.ontology_depth == "shallow":
        # More conservative promotion
        base = max(base, 0.65)
    elif guidance.ontology_depth == "deep":
        # More permissive promotion
        base = min(base, 0.35)

    # Apply concept promotion bias
    # Higher bias means easier promotion
    adjusted = base - (guidance.concept_promotion_bias - 0.5) * 0.30
    return max(0.0, min(1.0, adjusted))


def get_effective_hierarchy_threshold(guidance: UserGuidance | None) -> float:
    """
    Return the effective hierarchy threshold based on ontology depth.
    """
    if guidance is None:
        return 0.50

    base = guidance.hierarchy_min_confidence

    if guidance.ontology_depth == "shallow":
        # Fewer hierarchy links
        base = max(base, 0.70)
    elif guidance.ontology_depth == "deep":
        # More hierarchy links
        base = min(base, 0.35)

    return max(0.0, min(1.0, base))


def should_promote_confidence(confidence: float | None, guidance: UserGuidance | None) -> bool:
    """
    Decide whether a promotion is accepted based on effective threshold.
    """
    if confidence is None:
        # If no confidence is provided, accept only in balanced/deep mode
        if guidance is None:
            return True
        return guidance.ontology_depth != "shallow"

    return confidence >= get_effective_promotion_threshold(guidance)


def should_accept_hierarchy_confidence(confidence: float | None, guidance: UserGuidance | None) -> bool:
    """
    Decide whether a hierarchy link is accepted based on effective threshold.
    """
    if confidence is None:
        if guidance is None:
            return True
        return guidance.ontology_depth == "deep"

    return confidence >= get_effective_hierarchy_threshold(guidance)