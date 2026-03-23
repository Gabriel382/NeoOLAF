import sys
from pathlib import Path

from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner

from neoolaf.domain.documents import Document
from neoolaf.layers.layer00_preprocessing.component import PreprocessingLayer
from neoolaf.resources.ocr.paddle_engine import PaddleOCREngine

def run(pdf_path: str, use_chunking: bool = True):
    doc = Document(doc_id="doc_0001", source_path=pdf_path, raw_text="")
    state = PipelineState(document=doc, llm_model="none")
    state.artifact_dir = "exports"

    pipeline = Pipeline(layers=[
        PreprocessingLayer(
            chunk_size=1500,
            overlap=200,
            enable_chunking=use_chunking,
            ocr_engine=PaddleOCREngine(),
        ),
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


def run_folder(folder_path: str, use_chunking: bool = True):
    folder = Path(folder_path)
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {folder}")
        return []

    results = []
    failures = []

    for pdf_file in pdf_files:
        try:
            results.append(run(str(pdf_file), use_chunking=use_chunking))
        except Exception as exc:
            failures.append((pdf_file.name, str(exc)))
            print("=" * 60)
            print(f"  PDF:          {pdf_file}")
            print(f"  ERROR:        {exc}")
            print("=" * 60)

    print("\nSUMMARY:")
    print(f"  Processed:    {len(results)}")
    print(f"  Failed:       {len(failures)}")

    if failures:
        print("\nFAILURES:")
        for filename, error in failures:
            print(f"  {filename}: {error}")

    return results


if __name__ == "__main__":
    args = sys.argv[1:]
    use_chunking = "--no-chunking" not in args
    args = [arg for arg in args if arg != "--no-chunking"]

    target = args[0] if args else "data/XQuality/Textual"
    path = Path(target)

    if path.is_dir():
        run_folder(str(path), use_chunking=use_chunking)
    else:
        run(str(path), use_chunking=use_chunking)