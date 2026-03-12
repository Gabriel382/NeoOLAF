from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class UserGuidance:
    """
    User semantic guidance used to steer the semantic construction process.
    This object is optional but useful for domain-specific prompting.
    """
    domain_focus: Optional[str] = None
    abstraction_level: Optional[str] = None
    priority_relations: Optional[List[str]] = None
    population_policy: Optional[str] = None
    event_modeling_preference: Optional[str] = None