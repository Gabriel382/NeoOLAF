from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner

from neoolaf.domain.documents import Document
from neoolaf.domain.user_guidance import UserGuidance

from neoolaf.preprocessing.pdf_parsing import extract_text_from_pdf
from neoolaf.resources.llm_backends.ollama_backend import OllamaBackend

from neoolaf.layers.layer00_preprocessing.component import PreprocessingLayer
from neoolaf.layers.layer01_linguistic_expression_extraction.component import (
    LinguisticExpressionExtractionLayer,
)


def run_pdf_to_layer1(
    pdf_path: str,
    model_name: str = "gemma2:27b-instruct-q4_K_M",
):
    raw_text = extract_text_from_pdf(pdf_path)

    document = Document(
        doc_id="doc_0001",
        source_path=pdf_path,
        raw_text=raw_text,
    )

    guidance = UserGuidance(
        domain_focus="industrial maintenance and causal failure chains",
        abstraction_level="Treat machine types, failure types, and symptom types as concepts; treat document-specific occurrences as individuals.",
        priority_relations=["causal", "part-of", "affects", "observed-by", "temporal"],
        population_policy="Promote a candidate to concept only if it appears recurrently across documents.",
        event_modeling_preference="Treat failures, alarms, degradations, and shutdowns as events/states rather than simple entities.",
    )

    state = PipelineState(
        document=document,
        llm_model=model_name,
        user_guidance=guidance,
    )

    ollama = OllamaBackend()

    pipeline = Pipeline(
        layers=[
            PreprocessingLayer(chunk_size=1500, overlap=200),
            LinguisticExpressionExtractionLayer(
                ollama_backend=ollama,
                max_chunks=3,
                temperature=0.0,
            ),
        ]
    )

    runner = Runner(pipeline)
    final_state = runner.run(state)

    return final_state


if __name__ == "__main__":
    pdf_path = "your_textual_pdf.pdf"
    state = run_pdf_to_layer1(pdf_path)

    print("LOGS")
    for log in state.logs:
        print("-", log)

    print("\nEXPRESSIONS")
    for expr in state.linguistic_expressions:
        print(f"[{expr.expr_id}] {expr.text} | {expr.label}")
        print(f"  justification: {expr.justification}")
        print()