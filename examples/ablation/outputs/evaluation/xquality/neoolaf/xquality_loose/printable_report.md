# NeoOLAF Evaluation Report

## Main Metrics

| Target | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| Entity | 0.8929 | 0.1701 | 0.2857 | 50 | 6 | 244 |
| Relation | 1.0000 | 0.0843 | 0.1555 | 37 | 0 | 402 |

## Validation-Oriented Metrics

| STR | CR | PC | OC | CV | DVS |
| --- | --- | --- | --- | --- | --- |
| 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0811 | 1.0000 |

## Per-Relation Metrics

| relation | pred_count | gt_count | tp | fp | fn | precision | recall | f1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CAUSES | 6 | 105 | 6 | 0 | 99 | 1.0000 | 0.0571 | 0.1081 |
| HANDLED_BY | 4 | 96 | 4 | 0 | 92 | 1.0000 | 0.0417 | 0.0800 |
| REFERENCES | 0 | 23 | 0 | 0 | 23 | 0.0000 | 0.0000 | 0.0000 |
| REQUIRES | 15 | 132 | 15 | 0 | 117 | 1.0000 | 0.1136 | 0.2041 |
| TRIGGERS | 12 | 83 | 12 | 0 | 71 | 1.0000 | 0.1446 | 0.2526 |

## Ontology Metrics

| available | class_count | property_count | hierarchy_link_count | axiom_count | description_coverage | domain_coverage | range_coverage | duplicate_class_count | duplicate_property_count | hierarchy_depth | cycle_count | ontology_delta_size | promoted_concept_count | promoted_relation_count | ontology_growth_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| True | 69 | 69 | 68 | 0 | 1.0507 | 0.9855 | 0.9855 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0.0000 |

## Counts

- Total documents: 1
- Missing predictions: 0
- Parsed failures: 0
- Predicted entities: 56
- Gold entities: 294
- Predicted relations: 37
- Gold relations: 439