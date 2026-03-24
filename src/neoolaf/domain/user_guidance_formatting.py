from __future__ import annotations

# Local imports
from neoolaf.domain.user_guidance import UserGuidance


def build_user_guidance_context(
    guidance: UserGuidance | None,
    *,
    include_typing_examples: bool = False,
    include_relation_examples: bool = False,
    include_promotion_examples: bool = False,
    include_negative_examples: bool = False,
) -> str:
    """
    Build a compact prompt-ready textual context from UserGuidance.
    """
    if guidance is None:
        return ""

    lines = []

    # Base semantic instructions
    if guidance.domain_focus:
        lines.append(f"Domain focus: {guidance.domain_focus}")
    if guidance.abstraction_level:
        lines.append(f"Abstraction level: {guidance.abstraction_level}")
    if guidance.priority_relations:
        lines.append(f"Priority relations: {', '.join(guidance.priority_relations)}")
    if guidance.population_policy:
        lines.append(f"Population policy: {guidance.population_policy}")
    if guidance.event_modeling_preference:
        lines.append(f"Event modeling preference: {guidance.event_modeling_preference}")

    # Ontology depth
    lines.append(f"Ontology depth preference: {guidance.ontology_depth}")

    # Examples
    if include_typing_examples and guidance.typing_examples:
        lines.append("Typing examples:")
        for example in guidance.typing_examples:
            msg = f"- '{example.text}' -> {example.expected_type}"
            if example.explanation:
                msg += f" ({example.explanation})"
            lines.append(msg)

    if include_relation_examples and guidance.relation_examples:
        lines.append("Relation extraction examples:")
        for example in guidance.relation_examples:
            msg = (
                f"- '{example.text}' -> "
                f"({example.source_label}, {example.relation_label}, {example.target_label})"
            )
            if example.explanation:
                msg += f" ({example.explanation})"
            lines.append(msg)

    if include_promotion_examples and guidance.promotion_examples:
        lines.append("Promotion examples:")
        for example in guidance.promotion_examples:
            msg = f"- '{example.text}' -> promote={example.promote}"
            if example.promoted_label:
                msg += f", label={example.promoted_label}"
            if example.explanation:
                msg += f" ({example.explanation})"
            lines.append(msg)

    if include_negative_examples and guidance.negative_examples:
        lines.append("Negative examples:")
        for example in guidance.negative_examples:
            msg = f"- '{example.text}'"
            if example.target_layer:
                msg += f" [layer={example.target_layer}]"
            if example.explanation:
                msg += f" ({example.explanation})"
            lines.append(msg)

    if not lines:
        return ""

    return "\n".join(lines) + "\n\n"