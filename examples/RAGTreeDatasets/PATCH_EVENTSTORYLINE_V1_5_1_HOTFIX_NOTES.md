# EventStoryLine v1.5.1 guidance-loader hotfix

This patch fixes the pre-run `RelationExample.__init__()` error without changing
anything under `src/neoolaf`.

Corrections:

- converts compact `source/relation/target/why` relation examples to NeoOLAF's
  native `text/source_label/relation_label/target_label/explanation` schema;
- corrects `ontology_depth` to the supported `shallow` value;
- adds an experiment-side preflight normalizer and audit so future compact
  examples cannot crash the native loader;
- preserves the v1.5 notebook, profile, task guidance, prompts, pipeline and run
  directory.

After extracting, restart the notebook kernel before running from the first cell
so Python reloads the patched helper module.
