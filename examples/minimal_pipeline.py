import sys
from pathlib import Path

from neoolaf.core.pipeline import Pipeline
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.core.runner import Runner

from neoolaf.domain.documents import Document
from neoolaf.layers.layer00_preprocessing.component import PreprocessingLayer
from neoolaf.resources.translation.nllb_backend import NLLB200TranslatorBackend
from neoolaf.resources.translation.deep_translator_backend import DeepTranslatorBackend

def run(pdf_path: str):
    doc = Document(doc_id="doc_0001", source_path=pdf_path, raw_text="")
    state = PipelineState(document=doc, llm_model="none")
    state.artifact_dir = "exports"

    translator = DeepTranslatorBackend(max_chars=4000)

    pipeline = Pipeline(layers=[
        PreprocessingLayer(
            chunk_size=1500,
            overlap=200,
            #enable_chunking=False,
            translate=True,
            translator=translator,
            verbose=True,
        ),
    ])

    runner = Runner(pipeline)
    final = runner.run(state)

    print("=" * 60)
    print(f"  PDF:          {pdf_path}")
    print(f"  Type:         {final.document.pdf_type}")
    print(f"  Cleaned text: {len(final.document.cleaned_text)} chars")
    if final.document.translated_text:
        print(f"  Translated:   {len(final.document.translated_text)} chars")
    print(f"  Chunks:       {len(final.document.chunks)}")
    print("=" * 60)

    print("\nLOGS:")
    for log in final.logs:
        print(f"  {log}")

    print(f"\nCHUNK PREVIEWS:")
    for chunk in final.document.chunks[:3]:
        print(f"  [{chunk.chunk_id}] {chunk.text[:100]}...")

    return final


def run_folder(folder_path: str):
    folder = Path(folder_path)
    pdf_files = sorted(folder.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {folder}")
        return []

    results = []
    failures = []

    for pdf_file in pdf_files:
        try:
            results.append(run(str(pdf_file)))
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
    target = sys.argv[1] if len(sys.argv) > 1 else "data/XQuality/Textual"
    path = Path(target)

    if path.is_dir():
        run_folder(str(path))
    else:
        run(str(path))