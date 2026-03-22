from __future__ import annotations

# Standard library imports
from pathlib import Path

# Local imports
from neoolaf.core.base_layer import BaseLayer
from neoolaf.core.pipeline_state import PipelineState
from neoolaf.ontology.serialisation.ttl_serialiser import OntologyTTLSerialiser
from neoolaf.kg.serialisation.ttl_serialiser import KGTTLSerialiser
from neoolaf.kg.serialisation.json_serialiser import KGJSONSerialiser


class SerializationLayer(BaseLayer):
    """
    Layer 12: serialization / export.

    Responsibilities:
    - export local ontology
    - export inferred/completed ontology
    - export local KG
    - export inferred/completed KG
    """

    name = "layer12_serialization"

    def __init__(
        self,
        output_subdir: str = "exports",
        base_uri: str = "http://neoolaf.org/resource/",
        save_intermediate: bool = True,
        verbose: bool = False,
    ) -> None:
        """
        Initialize Layer 12.

        Args:
            output_subdir:
                Subdirectory inside the run folder where exports are written.
            base_uri:
                Base URI used by RDF serialisers.
            save_intermediate:
                Whether to save the layer artifact payload.
            verbose:
                Whether to print progress information.
        """
        super().__init__(save_intermediate=save_intermediate, verbose=verbose)
        self.output_subdir = output_subdir
        self.base_uri = base_uri

        self.ontology_serialiser = OntologyTTLSerialiser(base_uri=base_uri)
        self.kg_ttl_serialiser = KGTTLSerialiser(base_uri=base_uri)
        self.kg_json_serialiser = KGJSONSerialiser()

    def _run(self, state: PipelineState) -> PipelineState:
        """
        Serialize ontology and KG outputs into files.
        """
        if state.artifact_dir is None:
            raise ValueError("artifact_dir is required for serialization.")

        export_dir = Path(state.artifact_dir) / self.output_subdir
        export_dir.mkdir(parents=True, exist_ok=True)

        # Ontology
        ontology_local_path = export_dir / "ontology_local.ttl"
        ontology_inferred_path = export_dir / "ontology_inferred.ttl"

        # KG
        kg_local_ttl_path = export_dir / "kg_local.ttl"
        kg_inferred_ttl_path = export_dir / "kg_inferred.ttl"
        kg_local_json_path = export_dir / "kg_local.json"
        kg_inferred_json_path = export_dir / "kg_inferred.json"

        # Write ontology
        self.ontology_serialiser.serialise_local(state, str(ontology_local_path))
        self.ontology_serialiser.serialise_inferred(state, str(ontology_inferred_path))

        # Write KG
        self.kg_ttl_serialiser.serialise_local(state, str(kg_local_ttl_path))
        self.kg_ttl_serialiser.serialise_inferred(state, str(kg_inferred_ttl_path))
        self.kg_json_serialiser.serialise_local(state, str(kg_local_json_path))
        self.kg_json_serialiser.serialise_inferred(state, str(kg_inferred_json_path))

        state.log(f"[layer12_serialization] exports written to {export_dir}")

        if self.verbose:
            print(f"[NeoOLAF] Exports written to: {export_dir}")

        return state

    def build_artifact_payload(self, state: PipelineState) -> dict:
        """
        Serialize a lightweight artifact payload for Layer 12 itself.
        """
        return {
            "layer": self.name,
            "artifact_dir": state.artifact_dir,
            "output_subdir": self.output_subdir,
            "base_uri": self.base_uri,
        }