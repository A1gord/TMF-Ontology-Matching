import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any
import logging
from pathlib import Path

from ..entities.tmf_entity import TMFEntity, TMFOntology, EntityType


class TMFXMLParser:
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config

        default_tmf_namespace = "http://www.tmforum.org/ontology#"
        if config and hasattr(config, "parsing") and config.parsing.default_namespace:
            default_tmf_namespace = config.parsing.default_namespace

        self.namespace_map = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "owl": "http://www.w3.org/2002/07/owl#",
            "tmf": default_tmf_namespace,
        }

    def parse_file(self, file_path: str) -> TMFOntology:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            ontology_name = Path(file_path).stem
            ontology = TMFOntology(name=ontology_name)

            self._extract_namespace_info(root, ontology)
            self._parse_entities(root, ontology)
            self._parse_relationships(root, ontology)

            return ontology

        except ET.ParseError as e:
            self.logger.error(f"XML parsing error in {file_path}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Error parsing TMF file {file_path}: {e}")
            raise

    def _extract_namespace_info(self, root: ET.Element, ontology: TMFOntology) -> None:
        for prefix, uri in root.attrib.items():
            if prefix.startswith("xmlns:"):
                namespace_prefix = prefix[6:]
                self.namespace_map[namespace_prefix] = uri

        ontology.namespace = self.namespace_map.get("tmf", "")
        ontology.metadata["namespaces"] = self.namespace_map.copy()

    def _parse_entities(self, root: ET.Element, ontology: TMFOntology) -> None:
        for element in root.iter():
            entity = self._parse_entity_element(element)
            if entity:
                ontology.add_entity(entity)

    def _parse_entity_element(self, element: ET.Element) -> Optional[TMFEntity]:
        tag_name = self._get_local_name(element.tag)

        if tag_name in ["Class", "ObjectProperty", "DatatypeProperty", "Individual"]:
            entity = TMFEntity()

            entity.id = element.get(
                f'{{{self.namespace_map["rdf"]}}}about',
                element.get(f'{{{self.namespace_map["rdf"]}}}ID', ""),
            )

            if not entity.id:
                if (
                    self.config
                    and hasattr(self.config, "parsing")
                    and self.config.parsing.auto_generate_ids
                ):
                    entity.id = f"entity_{len(element.tag)}_{hash(element.tag) % 10000}"
                else:
                    return None

            entity.entity_type = self._map_element_to_entity_type(tag_name)

            for child in element:
                self._parse_entity_property(child, entity)

            return entity

        return None

    def _parse_entity_property(self, element: ET.Element, entity: TMFEntity) -> None:
        tag_name = self._get_local_name(element.tag)

        if tag_name == "label":
            entity.name = element.text or ""
        elif tag_name == "comment":
            entity.description = element.text or ""
        elif tag_name == "subClassOf":
            parent_ref = element.get(f'{{{self.namespace_map["rdf"]}}}resource')
            if parent_ref:
                entity.parent_id = parent_ref
        elif tag_name in ["domain", "range", "sameAs", "equivalentClass"]:
            resource_ref = element.get(f'{{{self.namespace_map["rdf"]}}}resource')
            if resource_ref:
                entity.add_relationship(resource_ref)
        else:
            if element.text:
                entity.add_property(tag_name, element.text)
            elif element.attrib:
                entity.add_property(tag_name, dict(element.attrib))

    def _parse_relationships(self, root: ET.Element, ontology: TMFOntology) -> None:
        for element in root.iter():
            if self._get_local_name(element.tag) == "ObjectProperty":
                self._parse_object_property(element, ontology)

    def _parse_object_property(
        self, element: ET.Element, ontology: TMFOntology
    ) -> None:
        property_id = element.get(
            f'{{{self.namespace_map["rdf"]}}}about',
            element.get(f'{{{self.namespace_map["rdf"]}}}ID', ""),
        )

        if not property_id:
            return

        domain_entities = []
        range_entities = []

        for child in element:
            tag_name = self._get_local_name(child.tag)
            resource_ref = child.get(f'{{{self.namespace_map["rdf"]}}}resource')

            if tag_name == "domain" and resource_ref:
                domain_entities.append(resource_ref)
            elif tag_name == "range" and resource_ref:
                range_entities.append(resource_ref)

        for domain_id in domain_entities:
            domain_entity = ontology.get_entity(domain_id)
            if domain_entity:
                for range_id in range_entities:
                    domain_entity.add_relationship(range_id)

    def _get_local_name(self, tag: str) -> str:
        if "}" in tag:
            return tag.split("}")[1]
        return tag

    def _map_element_to_entity_type(self, tag_name: str) -> EntityType:
        mapping = {
            "Class": EntityType.CONCEPT,
            "ObjectProperty": EntityType.RELATIONSHIP,
            "DatatypeProperty": EntityType.PROPERTY,
            "Individual": EntityType.CONCEPT,
        }
        return mapping.get(tag_name, EntityType.CONCEPT)


class TMFFormatParser:
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config

    def parse_file(self, file_path: str) -> TMFOntology:
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()

            ontology_name = Path(file_path).stem
            ontology = TMFOntology(name=ontology_name)

            lines = content.split("\n")
            current_entity = None

            for line in lines:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("Entity:"):
                    if current_entity:
                        ontology.add_entity(current_entity)
                    current_entity = self._parse_entity_declaration(line)
                elif current_entity and ":" in line:
                    self._parse_entity_attribute(line, current_entity)

            if current_entity:
                ontology.add_entity(current_entity)

            return ontology

        except Exception as e:
            self.logger.error(f"Error parsing TMF format file {file_path}: {e}")
            raise

    def _parse_entity_declaration(self, line: str) -> TMFEntity:
        parts = line.split(":", 1)
        if len(parts) < 2:
            raise ValueError(f"Invalid entity declaration: {line}")

        entity_info = parts[1].strip()
        entity = TMFEntity()

        if "(" in entity_info and ")" in entity_info:
            name_part = entity_info[: entity_info.index("(")]
            type_part = entity_info[entity_info.index("(") + 1 : entity_info.index(")")]

            entity.name = name_part.strip()
            entity.entity_type = self._parse_entity_type(type_part.strip())
        else:
            entity.name = entity_info

        entity.id = entity.name.replace(" ", "_").lower()

        return entity

    def _parse_entity_attribute(self, line: str, entity: TMFEntity) -> None:
        parts = line.split(":", 1)
        if len(parts) < 2:
            return

        attr_name = parts[0].strip().lower()
        attr_value = parts[1].strip()

        if attr_name == "description":
            entity.description = attr_value
        elif attr_name == "parent":
            entity.parent_id = attr_value.replace(" ", "_").lower()
        elif attr_name == "children":
            children = [
                child.strip().replace(" ", "_").lower()
                for child in attr_value.split(",")
            ]
            entity.children_ids.extend(children)
        elif attr_name == "synonyms":
            synonyms = [syn.strip() for syn in attr_value.split(",")]
            entity.synonyms.extend(synonyms)
        elif attr_name == "relationships":
            relationships = [
                rel.strip().replace(" ", "_").lower() for rel in attr_value.split(",")
            ]
            entity.relationships.extend(relationships)
        else:
            entity.add_property(attr_name, attr_value)

    def _parse_entity_type(self, type_str: str) -> EntityType:
        type_mapping = {
            "business_process": EntityType.BUSINESS_PROCESS,
            "information_model": EntityType.INFORMATION_MODEL,
            "functional_model": EntityType.FUNCTIONAL_MODEL,
            "component_model": EntityType.COMPONENT_MODEL,
            "concept": EntityType.CONCEPT,
            "property": EntityType.PROPERTY,
            "relationship": EntityType.RELATIONSHIP,
        }
        return type_mapping.get(type_str.lower(), EntityType.CONCEPT)


class UniversalTMFParser:
    def __init__(self, config=None):
        self.xml_parser = TMFXMLParser(config)
        self.tmf_parser = TMFFormatParser(config)
        self.logger = logging.getLogger(__name__)
        self.config = config

    def parse_file(self, file_path: str) -> TMFOntology:
        file_extension = Path(file_path).suffix.lower()

        try:
            if file_extension in [".xml", ".owl", ".rdf"]:
                return self.xml_parser.parse_file(file_path)
            elif file_extension in [".tmf", ".txt"]:
                return self.tmf_parser.parse_file(file_path)
            else:
                self.logger.warning(
                    f"Unknown file format: {file_extension}. Trying XML parser."
                )
                return self.xml_parser.parse_file(file_path)
        except Exception as e:
            self.logger.error(f"Failed to parse file {file_path}: {e}")
            raise
