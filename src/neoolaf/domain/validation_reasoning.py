from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional

# Local imports
from neoolaf.domain.candidate_triple import CandidateTriple
from neoolaf.domain.axiom_schema import AxiomSchemaCandidate
from neoolaf.domain.general_axiom import GeneralAxiomCandidate


@dataclass
class ValidationIssue:
    """
    One validation issue detected at document level.
    """

    # Stable identifier for the issue
    issue_id: str

    # Validation category
    issue_type: str

    # Severity level, for example: warning, error
    severity: str

    # Human-readable explanation
    message: str

    # Optional related artifact IDs
    related_ids: List[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    """
    Document-level validation result.
    """

    # Overall boolean status
    is_valid: bool

    # All detected issues
    issues: List[ValidationIssue] = field(default_factory=list)


@dataclass
class ReasoningReport:
    """
    Document-level reasoning result.
    """

    # Inferred graph assertions
    inferred_triples: List[CandidateTriple] = field(default_factory=list)

    # Inferred schema patterns or reused schema candidates
    inferred_axiom_schemata: List[AxiomSchemaCandidate] = field(default_factory=list)

    # Inferred general axioms
    inferred_general_axioms: List[GeneralAxiomCandidate] = field(default_factory=list)

    # Short textual notes
    notes: List[str] = field(default_factory=list)