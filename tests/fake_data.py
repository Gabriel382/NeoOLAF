"""
Fake data builders for evaluation testing.

Builds a realistic PipelineState and Gold standards
simulating a bearing failure maintenance report.
"""
from __future__ import annotations

from neoolaf.domain.documents import Document, DocumentChunk
from neoolaf.domain.linguistic_expression import Evidence
from neoolaf.domain.candidates import (
    EntityCandidate,
    RelationCandidate,
    AttributeCandidate,
    EventCandidate,
)
from neoolaf.domain.candidate_triple import CandidateTriple
from neoolaf.domain.ontology_elements import ConceptCandidate, OntologyRelationCandidate
from neoolaf.domain.hierarchy import ConceptHierarchyLink, RelationHierarchyLink
from neoolaf.domain.general_axiom import GeneralAxiomCandidate
from neoolaf.domain.validation_reasoning import (
    ValidationReport,
    ValidationIssue,
    ReasoningReport,
)
from neoolaf.domain.completion import CompletionCandidate
from neoolaf.core.pipeline_state import PipelineState

from neoolaf.evaluation.benchmark.entity_metrics import GoldEntity
from neoolaf.evaluation.benchmark.relation_metrics import GoldTriple
from neoolaf.evaluation.benchmark.ontology_conformance import GoldOntology
from neoolaf.evaluation.no_gold.ontology_alignment import ReferenceOntology


def _ev(snippet: str, chunk_id: str = "chunk-1") -> Evidence:
    """Helper to build a quick Evidence object."""
    return Evidence(
        chunk_id=chunk_id,
        chunk_start_char=0,
        chunk_end_char=len(snippet),
        doc_start_char=0,
        doc_end_char=len(snippet),
        snippet=snippet,
    )


def build_fake_state() -> PipelineState:
    """Build a realistic fake PipelineState with data in every layer."""

    doc = Document(
        doc_id="test-doc-001",
        source_path="/fake/path/bearing_report.pdf",
        raw_text=(
            "The bearing failure caused overheating in the motor assembly. "
            "Vibration analysis detected misalignment. The shaft seal was worn. "
            "Temperature sensors recorded abnormal readings above 120 degrees."
        ),
        pdf_type="textual",
        chunks=[
            DocumentChunk(chunk_id="chunk-1", text="The bearing failure caused overheating in the motor assembly.", start_char=0, end_char=61),
            DocumentChunk(chunk_id="chunk-2", text="Vibration analysis detected misalignment.", start_char=62, end_char=103),
            DocumentChunk(chunk_id="chunk-3", text="The shaft seal was worn. Temperature sensors recorded abnormal readings above 120 degrees.", start_char=104, end_char=194),
        ],
    )

    entities = [
        EntityCandidate(candidate_id="ent-1", canonical_label="BearingFailure", normalized_label="bearing failure", candidate_type="event", confidence=0.92),
        EntityCandidate(candidate_id="ent-2", canonical_label="Overheating", normalized_label="overheating", candidate_type="event", confidence=0.88),
        EntityCandidate(candidate_id="ent-3", canonical_label="MotorAssembly", normalized_label="motor assembly", candidate_type="component", confidence=0.95),
        EntityCandidate(candidate_id="ent-4", canonical_label="VibrationAnalysis", normalized_label="vibration analysis", candidate_type="process", confidence=0.85),
        EntityCandidate(candidate_id="ent-5", canonical_label="Misalignment", normalized_label="misalignment", candidate_type="symptom", confidence=0.80),
        EntityCandidate(candidate_id="ent-6", canonical_label="ShaftSeal", normalized_label="shaft seal", candidate_type="component", confidence=0.90),
        EntityCandidate(candidate_id="ent-7", canonical_label="TemperatureSensor", normalized_label="temperature sensor", candidate_type="component", confidence=0.87),
        EntityCandidate(candidate_id="ent-8", canonical_label="CoolingSystem", normalized_label="cooling system", candidate_type="component", confidence=0.60),
    ]

    relations = [
        RelationCandidate(candidate_id="rel-1", canonical_label="causes", normalized_label="causes", candidate_type="relation", confidence=0.90),
        RelationCandidate(candidate_id="rel-2", canonical_label="detects", normalized_label="detects", candidate_type="relation", confidence=0.85),
        RelationCandidate(candidate_id="rel-3", canonical_label="partOf", normalized_label="part of", candidate_type="relation", confidence=0.88),
    ]

    attributes = [
        AttributeCandidate(candidate_id="attr-1", canonical_label="temperature", normalized_label="temperature", candidate_type="attribute", confidence=0.82),
    ]

    events = [
        EventCandidate(candidate_id="evt-1", canonical_label="FailureEvent", normalized_label="failure event", candidate_type="event", confidence=0.91),
    ]

    triples = [
        CandidateTriple(
            triple_id="t-1", subject_id="ent-1", subject_label="BearingFailure", subject_type="event",
            predicate_id="rel-1", predicate_label="causes",
            object_id="ent-2", object_label="Overheating", object_type="event",
            chunk_id="chunk-1", justification="The bearing failure caused overheating in the motor assembly.",
            confidence=0.91,
            provenance=[_ev("The bearing failure caused overheating in the motor assembly.")],
        ),
        CandidateTriple(
            triple_id="t-2", subject_id="ent-4", subject_label="VibrationAnalysis", subject_type="process",
            predicate_id="rel-2", predicate_label="detects",
            object_id="ent-5", object_label="Misalignment", object_type="symptom",
            chunk_id="chunk-2", justification="Vibration analysis detected misalignment.",
            confidence=0.87,
            provenance=[_ev("Vibration analysis detected misalignment.", "chunk-2")],
        ),
        CandidateTriple(
            triple_id="t-3", subject_id="ent-6", subject_label="ShaftSeal", subject_type="component",
            predicate_id="rel-3", predicate_label="partOf",
            object_id="ent-3", object_label="MotorAssembly", object_type="component",
            chunk_id="chunk-1", justification="The shaft seal is part of the motor assembly.",
            confidence=0.85,
            provenance=[_ev("The shaft seal was worn.", "chunk-3")],
        ),
        CandidateTriple(
            triple_id="t-4", subject_id="ent-7", subject_label="TemperatureSensor", subject_type="component",
            predicate_id="rel-2", predicate_label="detects",
            object_id="ent-2", object_label="Overheating", object_type="event",
            chunk_id="chunk-3", justification="Temperature sensors recorded abnormal readings.",
            confidence=0.78,
            provenance=[_ev("Temperature sensors recorded abnormal readings above 120 degrees.", "chunk-3")],
        ),
        CandidateTriple(
            triple_id="t-5", subject_id="ent-1", subject_label="BearingFailure", subject_type="event",
            predicate_id="rel-1", predicate_label="causes",
            object_id="ent-5", object_label="Misalignment", object_type="symptom",
            chunk_id="chunk-2", justification="Bearing failure causes misalignment.",
            confidence=0.65,
            provenance=[_ev("Vibration analysis detected misalignment.", "chunk-2")],
        ),
    ]

    concepts = [
        ConceptCandidate(concept_id="c-1", label="FailureEvent", normalized_label="failure event", concept_kind="event", confidence=0.90, justification="Promoted from entity candidates"),
        ConceptCandidate(concept_id="c-2", label="MechanicalComponent", normalized_label="mechanical component", concept_kind="component", confidence=0.88, justification="Promoted from entity candidates"),
        ConceptCandidate(concept_id="c-3", label="Symptom", normalized_label="symptom", concept_kind="symptom", confidence=0.85, justification="Promoted from entity candidates"),
        ConceptCandidate(concept_id="c-4", label="DiagnosticProcess", normalized_label="diagnostic process", concept_kind="process", confidence=0.82, justification="Promoted from entity candidates"),
        ConceptCandidate(concept_id="c-5", label="BearingDefect", normalized_label="bearing defect", concept_kind="failure", confidence=0.87, justification="Promoted from entity candidates"),
    ]

    onto_relations = [
        OntologyRelationCandidate(relation_id="or-1", label="causes", normalized_label="causes", confidence=0.90, justification="Promoted from relation candidates"),
        OntologyRelationCandidate(relation_id="or-2", label="detects", normalized_label="detects", confidence=0.85, justification="Promoted from relation candidates"),
        OntologyRelationCandidate(relation_id="or-3", label="partOf", normalized_label="part of", confidence=0.88, justification="Promoted from relation candidates"),
    ]

    concept_hier = [
        ConceptHierarchyLink(link_id="ch-1", child_concept_id="c-5", child_label="BearingDefect", parent_concept_id="c-1", parent_label="FailureEvent", justification="BearingDefect is a subclass of FailureEvent", confidence=0.88),
        ConceptHierarchyLink(link_id="ch-2", child_concept_id="c-3", child_label="Symptom", parent_concept_id="c-1", parent_label="FailureEvent", justification="Symptom relates to FailureEvent", confidence=0.75),
    ]

    relation_hier = [
        RelationHierarchyLink(link_id="rh-1", child_relation_id="or-2", child_label="detects", parent_relation_id="or-1", parent_label="causes", justification="detects is a diagnostic variant", confidence=0.70),
    ]

    axioms = [
        GeneralAxiomCandidate(
            axiom_id="ax-1", axiom_type="relation_domain", subject_id="or-1", subject_label="causes",
            predicate="rdfs:domain", object_id="c-1", object_label="FailureEvent",
            justification="causes has domain FailureEvent", confidence=0.88,
            evidence=[_ev("The bearing failure caused overheating.")],
        ),
        GeneralAxiomCandidate(
            axiom_id="ax-2", axiom_type="relation_range", subject_id="or-1", subject_label="causes",
            predicate="rdfs:range", object_id="c-3", object_label="Symptom",
            justification="causes has range Symptom", confidence=0.85,
            evidence=[_ev("The bearing failure caused overheating.")],
        ),
        GeneralAxiomCandidate(
            axiom_id="ax-3", axiom_type="description", subject_id="c-5", subject_label="BearingDefect",
            predicate="rdfs:comment",
            literal_value="A defect in a bearing component leading to mechanical failure.",
            justification="Description axiom for BearingDefect", confidence=0.80,
            evidence=[_ev("The bearing failure caused overheating in the motor assembly.")],
        ),
        GeneralAxiomCandidate(
            axiom_id="ax-4", axiom_type="subclass", subject_id="c-5", subject_label="BearingDefect",
            predicate="rdfs:subClassOf", object_id="c-1", object_label="FailureEvent",
            justification="BearingDefect is a subclass of FailureEvent", confidence=0.90,
            evidence=[_ev("The bearing failure caused overheating.")],
        ),
    ]

    validation_report = ValidationReport(
        is_valid=False,
        issues=[
            ValidationIssue(issue_id="vi-1", issue_type="missing_domain", severity="warning", message="Relation 'detects' has no domain axiom"),
            ValidationIssue(issue_id="vi-2", issue_type="missing_range", severity="warning", message="Relation 'detects' has no range axiom"),
            ValidationIssue(issue_id="vi-3", issue_type="orphan_concept", severity="warning", message="Concept 'DiagnosticProcess' has no parent"),
            ValidationIssue(issue_id="vi-4", issue_type="low_confidence", severity="error", message="Triple t-5 has low confidence (0.65)"),
            ValidationIssue(issue_id="vi-5", issue_type="contradiction", severity="error", message="Contradiction between t-1 and t-5"),
        ],
    )

    inferred_triple = CandidateTriple(
        triple_id="t-inf-1", subject_id="ent-5", subject_label="Misalignment", subject_type="symptom",
        predicate_id="rel-1", predicate_label="causes",
        object_id="ent-2", object_label="Overheating", object_type="event",
        chunk_id="chunk-2", justification="Inferred: Misalignment causes Overheating (transitive)",
        confidence=0.70,
        provenance=[_ev("Vibration analysis detected misalignment.", "chunk-2")],
    )

    reasoning_report = ReasoningReport(
        inferred_triples=[inferred_triple],
        notes=["Transitive closure applied on causes relation"],
    )

    completions = [
        CompletionCandidate(
            completion_id="comp-1", completion_type="graph_completion",
            justification="Added missing partOf relation for TemperatureSensor",
            confidence=0.75,
        ),
        CompletionCandidate(
            completion_id="comp-2", completion_type="ontology_completion",
            justification="Added range axiom for detects relation",
            confidence=0.80,
        ),
    ]

    return PipelineState(
        document=doc,
        llm_model="gpt-4o-test",
        entity_candidates=entities,
        relation_candidates=relations,
        attribute_candidates=attributes,
        event_candidates=events,
        candidate_triples=triples,
        concept_candidates=concepts,
        ontology_relation_candidates=onto_relations,
        concept_hierarchy_links=concept_hier,
        relation_hierarchy_links=relation_hier,
        general_axiom_candidates=axioms,
        validation_report=validation_report,
        reasoning_report=reasoning_report,
        completion_candidates=completions,
    )


def build_gold_entities() -> list[GoldEntity]:
    """Gold entities for benchmark evaluation."""
    return [
        GoldEntity(label="BearingFailure", entity_type="event", aliases=["Bearing Failure", "bearing defect"]),
        GoldEntity(label="Overheating", entity_type="event", aliases=["over-heating"]),
        GoldEntity(label="MotorAssembly", entity_type="component", aliases=["Motor Assembly", "motor"]),
        GoldEntity(label="VibrationAnalysis", entity_type="process", aliases=["Vibration Analysis"]),
        GoldEntity(label="Misalignment", entity_type="symptom"),
        GoldEntity(label="ShaftSeal", entity_type="component", aliases=["Shaft Seal"]),
        GoldEntity(label="TemperatureSensor", entity_type="component", aliases=["Temperature Sensor"]),
        GoldEntity(label="LubricationPump", entity_type="component"),
    ]


def build_gold_triples() -> list[GoldTriple]:
    """Gold triples for benchmark evaluation."""
    return [
        GoldTriple(subject="BearingFailure", predicate="causes", object="Overheating"),
        GoldTriple(subject="VibrationAnalysis", predicate="detects", object="Misalignment"),
        GoldTriple(subject="ShaftSeal", predicate="partOf", object="MotorAssembly"),
        GoldTriple(subject="TemperatureSensor", predicate="monitors", object="MotorAssembly"),
    ]


def build_gold_ontology() -> GoldOntology:
    """Gold ontology for benchmark evaluation."""
    return GoldOntology(
        concepts={"FailureEvent", "MechanicalComponent", "Symptom", "DiagnosticProcess", "BearingDefect", "LubricationSystem"},
        relations={"causes", "detects", "partOf", "monitors"},
        hierarchy=[
            ("BearingDefect", "FailureEvent"),
            ("Symptom", "FailureEvent"),
        ],
        domain_range=[
            ("causes", "FailureEvent", "Symptom"),
        ],
    )


def build_reference_ontology() -> ReferenceOntology:
    """Reference ontology for no-gold alignment evaluation."""
    ref = ReferenceOntology(
        concepts={"FailureMode", "MechanicalPart", "Symptom", "AnalysisProcess", "BearingFault"},
        relations={"causeOf", "identifies", "isPartOf"},
        hierarchy=[
            ("BearingFault", "FailureMode"),
            ("Symptom", "FailureMode"),
        ],
    )
    ref._norm_concepts = {c.lower().strip().replace("_", " ").replace("-", " ") for c in ref.concepts}
    ref._norm_relations = {r.lower().strip().replace("_", " ").replace("-", " ") for r in ref.relations}
    return ref
