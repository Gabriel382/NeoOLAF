from __future__ import annotations

# Dataclass utilities
from dataclasses import dataclass, field
from typing import List, Optional

# Local imports
from neoolaf.domain.documents import Document
from neoolaf.domain.linguistic_expression import LinguisticExpression
from neoolaf.domain.enriched_expression import EnrichedExpression
from neoolaf.domain.user_guidance import UserGuidance
from neoolaf.domain.candidates import (
    EntityCandidate,
    RelationCandidate,
    AttributeCandidate,
    EventCandidate,
)


from neoolaf.domain.relation_assertion import CandidateRelationAssertion
from neoolaf.domain.candidate_triple import CandidateTriple
from neoolaf.domain.ontology_elements import ConceptCandidate, OntologyRelationCandidate
from neoolaf.domain.hierarchy import ConceptHierarchyLink, RelationHierarchyLink
from neoolaf.domain.axiom_schema import AxiomSchemaCandidate
from neoolaf.domain.general_axiom import GeneralAxiomCandidate
from neoolaf.domain.validation_reasoning import ValidationReport, ReasoningReport
from neoolaf.domain.completion import CompletionCandidate
from neoolaf.domain.seed_ontology import SeedOntology

@dataclass
class PipelineState:
    """
    Shared pipeline state passed through all NeoOLAF layers.
    """

    # Current document being processed
    document: Document

    # LLM model name used by the pipeline
    llm_model: str

    # Optional semantic guidance
    user_guidance: Optional[UserGuidance] = None

    # Optional seed/source ontology
    seed_ontology: Optional[SeedOntology] = None

    # Directory where intermediate artifacts are stored
    artifact_dir: Optional[str] = None

    # Layer 1 outputs
    linguistic_expressions: List[LinguisticExpression] = field(default_factory=list)

    # Layer 2 outputs
    enriched_expressions: List[EnrichedExpression] = field(default_factory=list)

    # Layer 3 outputs
    entity_candidates: List[EntityCandidate] = field(default_factory=list)
    relation_candidates: List[RelationCandidate] = field(default_factory=list)
    attribute_candidates: List[AttributeCandidate] = field(default_factory=list)
    event_candidates: List[EventCandidate] = field(default_factory=list)

    # Layer 4 outputs
    candidate_relation_assertions: List[CandidateRelationAssertion] = field(default_factory=list)

    # Layer 5 outputs
    candidate_triples: List[CandidateTriple] = field(default_factory=list)

    # Layer 6 outputs
    concept_candidates: List[ConceptCandidate] = field(default_factory=list)
    ontology_relation_candidates: List[OntologyRelationCandidate] = field(default_factory=list)

    # Layer 7 outputs
    concept_hierarchy_links: List[ConceptHierarchyLink] = field(default_factory=list)
    relation_hierarchy_links: List[RelationHierarchyLink] = field(default_factory=list)

    # Layer 8 outputs
    axiom_schema_candidates: List[AxiomSchemaCandidate] = field(default_factory=list)

    # Layer 9 outputs
    general_axiom_candidates: List[GeneralAxiomCandidate] = field(default_factory=list)

    # Layer 10 outputs
    validation_report: ValidationReport | None = None
    reasoning_report: ReasoningReport | None = None

    # Layer 11 outputs
    completion_candidates: List[CompletionCandidate] = field(default_factory=list)

    # Execution logs
    logs: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """
        Append a message to the execution log.
        """
        self.logs.append(message)