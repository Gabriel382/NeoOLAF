import os
import re
import time
import sys
import logging
import json

import nltk
import spacy
import openpyxl
from openpyxl import Workbook
import csv
import gc

from openai import OpenAI

from olaf.pipeline.pipeline_schema import Pipeline
from olaf.repository.corpus_loader.text_corpus_loader import TextCorpusLoader
from olaf.pipeline.data_preprocessing.token_selector_data_preprocessing import TokenSelectorDataPreprocessing
from olaf.commons.spacy_processing_tools import is_not_stopword, is_not_punct
from olaf.commons.llm_tools import LLMGenerator
from olaf.pipeline.pipeline_component.term_extraction.llm_term_extraction import LLMTermExtraction
from olaf.pipeline.pipeline_component.candidate_term_enrichment.llm_based_enrichment import LLMBasedTermEnrichment
from olaf.pipeline.pipeline_component.concept_relation_extraction.llm_based_concept_extraction import LLMBasedConceptExtraction
from olaf.pipeline.pipeline_component.concept_relation_extraction.llm_based_relation_extraction import LLMBasedRelationExtraction
from olaf.pipeline.pipeline_component.axiom_extraction.llm_based_axiom_extraction import LLMBasedOWLAxiomExtraction
from olaf.commons.prompts import (
    openai_prompt_concept_term_extraction,
    openai_prompt_relation_term_extraction,
    openai_prompt_term_enrichment,
    openai_prompt_concept_extraction,
    openai_prompt_relation_extraction,
    openai_prompt_owl_axiom_extraction,
)

from olaf.repository.serialiser.rdf_owl_serialisers.base_owl_serialiser import BaseOWLSerialiser


# Simple corpus loader for in-memory batches
class BatchCorpusLoader:
    """A corpus loader that returns a pre-loaded batch of documents."""
    def __init__(self, documents):
        self.documents = documents
    
    def load_corpus(self):
        return self.documents


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
logger.addHandler(stream_handler)


def _load_resume_state(state_path: str):
    if not os.path.exists(state_path):
        return {"processed_split_files": [], "docs_processed": 0}
    try:
        with open(state_path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        if not isinstance(state, dict):
            raise ValueError("invalid state file")
        state.setdefault("processed_split_files", [])
        state.setdefault("docs_processed", 0)
        return state
    except Exception:
        logger.exception("Failed to load resume state; starting fresh")
        return {"processed_split_files": [], "docs_processed": 0}


def _save_resume_state(state_path: str, state: dict):
    tmp_path = state_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(state, fh, ensure_ascii=False, indent=2)
    os.replace(tmp_path, state_path)


def _clean_output(text: str) -> str:
    text = re.sub(r"```\w*", "", text).strip()
    text = (text.replace("“", '"').replace("”", '"')
                .replace("‘", "'").replace("’", "'"))

    if text.lstrip().startswith(("@prefix", "@base", "<http")):
        return text

    brace_pos   = text.find('{')
    bracket_pos = text.find('[')

    if bracket_pos != -1 and (brace_pos == -1 or bracket_pos < brace_pos):
        order = [('[', ']'), ('{', '}')]
    else:
        order = [('{', '}'), ('[', ']')]

    for open_ch, close_ch in order:
        start = text.find(open_ch)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            depth += (ch == open_ch) - (ch == close_ch)
            if depth == 0:
                return text[start:i + 1]
    return text


class LocalLLMGenerator(LLMGenerator):
    """vLLM-backed generator with retry/backoff for transient failures."""

    def check_resources(self):
        pass

    def generate_text(self, prompt):
        msgs = ([{"role": "user", "content": prompt}]
                if isinstance(prompt, str) else prompt)
        last_exc = None
        for attempt in range(4):
            try:
                # log prompt size and a short snippet for debugging
                try:
                    if isinstance(prompt, str):
                        logger.info("LLM prompt length=%d snippet=%s", len(prompt), prompt[:300])
                    else:
                        # compute approximate total content length
                        total_len = sum(len(m.get('content','')) for m in msgs)
                        logger.info("LLM prompt total messages=%d total_content_len=%d snippet=%s", len(msgs), total_len, str(msgs)[:300])
                except Exception:
                    logger.exception("Failed to log prompt info")

                raw = (OpenAI()
                       .chat.completions
                       .create(model=_MODEL, messages=msgs)
                       .choices[0].message.content or "")
                # log raw response for debugging (length + small preview)
                try:
                    logger.info("LLM raw response length=%d", len(raw) if raw is not None else 0)
                    logger.info("LLM raw response preview=%s", (raw[:1000] if raw else repr(raw)))
                except Exception:
                    logger.exception("Failed to log raw LLM response")
                cleaned = _clean_output(raw)
                if cleaned:
                    return cleaned
                raise ValueError("Empty response from model")
            except Exception as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning("  [retry %d/4 after %ds] %s", attempt+1, wait, exc)
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after 4 attempts: {last_exc}")


def main():
        os.environ["JAVA_EXE"]  = "/usr/bin/java"
        os.environ["ROBOT_JAR"] = "/home/selassri/tools/robot/robot.jar"
        os.environ["DATA_PATH"] = "/home/selassri/Documents/OLAF_DEMO/output"
        os.makedirs(os.environ["DATA_PATH"], exist_ok=True)

        os.makedirs("output", exist_ok=True)
        log_file_path = os.path.join("output", f"olaf_llm_{time.strftime('%Y%m%d_%H%M%S')}.log")
        # open a line-buffered file stream so writes appear promptly
        file_stream = open(log_file_path, mode="a", buffering=1, encoding="utf-8")
        file_handler = logging.StreamHandler(file_stream)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s: %(message)s"))
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file_path}")

        # helper to capture prints to logger
        class StreamToLogger:
            def __init__(self, logger, level=logging.INFO):
                self.logger = logger
                self.level = level

            def write(self, buf):
                for line in buf.rstrip().splitlines():
                    self.logger.log(self.level, line)

            def flush(self):
                pass

        # redirect stdout/stderr to logger so print() ends up in the log file
        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = StreamToLogger(logger, logging.INFO)
        sys.stderr = StreamToLogger(logger, logging.ERROR)

        try:
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)

            nlp = spacy.load("en_core_web_lg")

            os.environ["OPENAI_API_KEY"]  = "dummy"
            os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
            global _MODEL
            _MODEL = "openai/gpt-oss-20b"

            llm = LocalLLMGenerator()

            # Look for split files in common output dirs or matching pattern
            split_candidates = []
            # specific dirs we created earlier
            for d in ("data/DocRED/split_txt"):
                if os.path.isdir(d):
                    split_candidates.extend([os.path.join(d, f) for f in os.listdir(d)])

            split_files = sorted({os.path.abspath(p) for p in split_candidates if os.path.isfile(p)})

            # We'll avoid accumulating all results in memory.
            # Instead write per-split results to disk and append to cumulative files.
            cumulative_csv = os.path.join("output", "cumulative_llm_docred_triplets.csv")
            cumulative_ttl = os.path.join("output", "cumulative_llm_docred_ontology.ttl")
            resume_state_path = os.path.join("output", "llm_docred_resume_state.json")
            resume_state = _load_resume_state(resume_state_path)
            processed_split_files = set(os.path.abspath(p) for p in resume_state.get("processed_split_files", []))
            docs_processed = int(resume_state.get("docs_processed", 0) or 0)

            logger.info(
                "Resume state loaded: docs_processed=%s, processed_split_files=%d",
                docs_processed,
                len(processed_split_files),
            )

            for idx, split_path in enumerate(split_files, start=1):
                if split_path in processed_split_files:
                    logger.info(f"Skipping already processed split file: {split_path}")
                    print(f"Skipping already processed split file: {split_path}")
                    continue
                logger.info(f"Processing split file: {split_path}")
                print(f"\n=== Processing split file {idx}/{len(split_files)}: {split_path} ===")
                batch_pipeline = Pipeline(
                    spacy_model=nlp,
                    corpus_loader=TextCorpusLoader(corpus_path=split_path),
                    seed_kr=None,
                    preprocessing_components=[
                        TokenSelectorDataPreprocessing(
                            selector=lambda t: is_not_stopword(t) and is_not_punct(t),
                        )
                    ],
                    pipeline_components=[],
                )
                batch_pipeline.run()

            num_docs_in_batch = len(batch_pipeline.corpus)
            docs_processed += num_docs_in_batch
            cumulative_count = docs_processed

            logger.info(f"Split {idx}: documents={num_docs_in_batch}, cumulative={cumulative_count}")

            # Run pipeline phases on this batch_pipeline
            logger.info(f"Phase 1 — extract concept candidate terms")
            LLMTermExtraction(
                    prompt_template=openai_prompt_concept_term_extraction,
                llm_generator=llm,
            ).run(batch_pipeline)

            logger.info(f"Phase 2 — enrich candidate terms")
            LLMBasedTermEnrichment(
                prompt_template=openai_prompt_term_enrichment,
                llm_generator=llm,
            ).run(batch_pipeline)

            logger.info(f"Phase 3 — group terms into concepts")
            LLMBasedConceptExtraction(
                prompt_template=openai_prompt_concept_extraction,
                llm_generator=llm,
            ).run(batch_pipeline)

            logger.info(f"Phase 4 — extract relation candidate terms")
            LLMTermExtraction(
                prompt_template=openai_prompt_relation_term_extraction,
                llm_generator=llm,
            ).run(batch_pipeline)

            logger.info(f"Phase 5 — extract relations between concepts")
            LLMBasedRelationExtraction(
                prompt_template=openai_prompt_relation_extraction,
                llm_generator=llm,
                concept_max_distance=11,
            ).run(batch_pipeline)

            logger.info(f"Phase 6 — generate OWL axioms via LLM")
            LLMBasedOWLAxiomExtraction(
                prompt_template=openai_prompt_owl_axiom_extraction,
                llm_generator=llm,
                namespace="http://semanticweb.org/STEaMINg/DocREDOntology#",
            ).run(batch_pipeline)

            # Per-split results (do not keep them globally)
            batch_concepts = getattr(batch_pipeline.kr, "concepts", []) or []
            batch_relations = getattr(batch_pipeline.kr, "relations", []) or []

            # Write per-split CSV of triplets
            per_split_csv = os.path.join("output", f"llm_docred_triplets_part_{idx:03d}.csv")
            os.makedirs(os.path.dirname(per_split_csv), exist_ok=True)
            with open(per_split_csv, "w", encoding="utf-8", newline="") as fh:
                # header
                fh.write("Source Node,Target Node,Relation\n")
                for r in batch_relations:
                    src = r.source_concept.label if r.source_concept else "?"
                    dst = r.destination_concept.label if r.destination_concept else "?"
                    # escape double-quotes by doubling them for CSV
                    src_e = src.replace('"', '""')
                    dst_e = dst.replace('"', '""')
                    lbl_e = (r.label or "").replace('"', '""')
                    line = f'"{src_e}","{dst_e}","{lbl_e}"\n'
                    fh.write(line)
            logger.info(f"Wrote per-split triplets -> {per_split_csv}")

            # Append per-split triplets to cumulative CSV (create header if missing)
            write_header = not os.path.exists(cumulative_csv)
            with open(cumulative_csv, "a", encoding="utf-8", newline="") as fh:
                if write_header:
                    fh.write("Source Node,Target Node,Relation\n")
                for r in batch_relations:
                    src = r.source_concept.label if r.source_concept else "?"
                    dst = r.destination_concept.label if r.destination_concept else "?"
                    src_e = src.replace('"', '""')
                    dst_e = dst.replace('"', '""')
                    lbl_e = (r.label or "").replace('"', '""')
                    line = f'"{src_e}","{dst_e}","{lbl_e}"\n'
                    fh.write(line)

            # Export per-split TTL for the batch and append to cumulative TTL
            per_split_ttl = os.path.join("output", f"llm_docred_ontology_part_{idx:03d}.ttl")
            owl_serialiser = BaseOWLSerialiser(
                base_uri="http://semanticweb.org/STEaMINg/DocREDOntology#",
                keep_all_labels=True,
            )
            owl_serialiser.build_graph(batch_pipeline.kr)
            owl_serialiser.export_graph(file_path=per_split_ttl, rdf_format="turtle")
            logger.info(f"Wrote per-split TTL -> {per_split_ttl}")

            # Append TTL content to cumulative TTL, skipping duplicate prefix/base lines after first file
            prefix_re = re.compile(r"^(@prefix|@base).*")
            if not os.path.exists(cumulative_ttl):
                # first file -> copy entire content
                with open(per_split_ttl, "r", encoding="utf-8") as src, open(cumulative_ttl, "w", encoding="utf-8") as dst:
                    dst.write(src.read())
            else:
                # append but skip prefix/base lines
                with open(per_split_ttl, "r", encoding="utf-8") as src, open(cumulative_ttl, "a", encoding="utf-8") as dst:
                    for line in src:
                        if prefix_re.match(line.strip()):
                            continue
                        dst.write(line)

            logger.info(f"Appended per-split TTL to cumulative TTL -> {cumulative_ttl}")

            # Produce cumulative XLSX snapshot by streaming from cumulative CSV -> avoid loading all relations
            try:
                wb = Workbook(write_only=True)
                ws = wb.create_sheet()
                with open(cumulative_csv, "r", encoding="utf-8", newline="") as csvf:
                    reader = csv.reader(csvf)
                    for row in reader:
                        ws.append(row)
                snapshot_xlsx = os.path.join("output", f"llm_docred_triplets_{cumulative_count}.xlsx")
                wb.save(snapshot_xlsx)
                logger.info(f"Wrote cumulative XLSX snapshot -> {snapshot_xlsx}")
                print(f"Snapshot XLSX up to {cumulative_count} docs -> {snapshot_xlsx}")
            except Exception:
                logger.exception("Failed to write cumulative XLSX snapshot")

            # Print per-split summary
            print(f"Split {idx}: wrote {len(batch_relations)} triplets, {len(batch_concepts)} concepts")
            logger.info(f"Split {idx}: wrote {len(batch_relations)} triplets, {len(batch_concepts)} concepts")

            # Free memory from this batch
            try:
                del batch_pipeline
                del batch_relations
                del batch_concepts
            except Exception:
                pass
            gc.collect()

            if split_path != "__inline_full__":
                processed_split_files.add(os.path.abspath(split_path))
                resume_state["processed_split_files"] = sorted(processed_split_files)
                resume_state["docs_processed"] = docs_processed
                _save_resume_state(resume_state_path, resume_state)
                logger.info("Resume state saved after split %s", split_path)

            logger.info("=== All split files processed ===")
            print(f"\n=== All split files processed ===")
            # Compute final totals from cumulative CSV if present (avoid holding all data in memory)
            if os.path.exists(cumulative_csv):
                try:
                    import csv
                    concepts_set = set()
                    rel_count = 0
                    with open(cumulative_csv, "r", encoding="utf-8", newline="") as csvf:
                        reader = csv.reader(csvf)
                        first = True
                        for row in reader:
                            if first:
                                first = False
                                continue
                            if not row:
                                continue
                            rel_count += 1
                            concepts_set.add(row[0])
                            concepts_set.add(row[1])
                    print(f"Final totals: {len(concepts_set)} concepts, {rel_count} relations")
                    logger.info(f"Final totals: {len(concepts_set)} concepts, {rel_count} relations")
                except Exception:
                    logger.exception("Failed to compute final totals from cumulative CSV")
            else:
                print(f"Final totals: processed {docs_processed} documents (no cumulative CSV found)")
                logger.info(f"Final totals: processed {docs_processed} documents (no cumulative CSV found)")

        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr
            try:
                logging.shutdown()
            finally:
                try:
                    file_stream.close()
                except Exception:
                    pass


if __name__ == "__main__":
    main()