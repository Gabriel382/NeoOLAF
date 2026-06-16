"""Prompt templates adapted from the original TaxoDrivenKG style."""

PROMPT_TEMPLATE = """
-Goal-
Given a text document and a list of potential entities from a domain taxonomy, identify all entities and relationships that are explicitly supported by the text.

-Steps-
1. Extract all named or domain-relevant entities in the text. For each entity, output:
   - entity_name
   - entity_type
   - entity_description

2. Extract all explicit relationships between the extracted entities. For each relationship, output:
   - source_entity
   - target_entity
   - relationship_type

3. Return the answer in English as a single list of records separated by **{record_delimiter}**.

Use exactly these formats:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>)

Potential entity candidates from the ontology: {potential_entities}

######################
{section_delimiter}Examples{section_delimiter}
{formatted_examples}
######################
{section_delimiter}Real Data{section_delimiter}
######################
Text: {input_text}
######################
Output:
"""

PROMPT_TEMPLATE_ZERO_SHOT = """
-Goal-
Given a text document and a list of potential entities from a domain taxonomy, identify all entities and relationships that are explicitly supported by the text.

Return the answer in English as a single list of records separated by **{record_delimiter}**.

Use exactly these formats:
("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)
("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>)

Potential entity candidates from the ontology: {potential_entities}
Text: {input_text}
Output:
"""

PROMPT_TEMPLATE_NO_RAG = PROMPT_TEMPLATE_ZERO_SHOT.replace(
    "Potential entity candidates from the ontology: {potential_entities}\n", ""
)
