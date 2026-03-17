import sys

from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner

from neoolaf.domain.documents import Document
from neoolaf.layers.layer00_preprocessing.component import PreprocessingLayer


def run(pdf_path: str):
    doc = Document(doc_id="doc_0001", source_path=pdf_path, raw_text="")
    state = PipelineState(document=doc, llm_model="none")

    pipeline = Pipeline(layers=[
        PreprocessingLayer(chunk_size=1500, overlap=200),
    ])

    runner = Runner(pipeline)
    final = runner.run(state)

    print("=" * 60)
    print(f"  PDF:          {pdf_path}")
    print(f"  Type:         {final.document.pdf_type}")
    print(f"  Cleaned text: {len(final.document.cleaned_text)} chars")
    print(f"  Chunks:       {len(final.document.chunks)}")
    print("=" * 60)

    print("\nLOGS:")
    for log in final.logs:
        print(f"  {log}")

    print(f"\nCHUNK PREVIEWS:")
    for chunk in final.document.chunks[:3]:
        print(f"  [{chunk.chunk_id}] {chunk.text[:100]}...")

    return final


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "data/XQuality/Textual/Chapitre_8_Alarmes_et_messages.pdf"
    run(pdf)
