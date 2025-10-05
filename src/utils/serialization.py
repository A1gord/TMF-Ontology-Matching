import json
import xml.etree.ElementTree as ET
from typing import Dict, Any, List
import logging
from pathlib import Path

from ..entities.tmf_entity import TMFOntology, TMFEntity


class JSONSerializer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def serialize_ontology(self, ontology: TMFOntology, file_path: str) -> None:
        try:
            data = ontology.to_dict()
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=2, ensure_ascii=False)
            self.logger.info(f"Ontology serialized to JSON: {file_path}")
        except Exception as e:
            self.logger.error(f"Error serializing ontology to JSON: {e}")
            raise
    
    def deserialize_ontology(self, file_path: str) -> TMFOntology:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            ontology = TMFOntology.from_dict(data)
            self.logger.info(f"Ontology deserialized from JSON: {file_path}")
            return ontology
        except Exception as e:
            self.logger.error(f"Error deserializing ontology from JSON: {e}")
            raise
    
    def serialize_matching_results(self, results: Dict[str, Any], file_path: str) -> None:
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(results, file, indent=2, ensure_ascii=False)
            self.logger.info(f"Matching results serialized to JSON: {file_path}")
        except Exception as e:
            self.logger.error(f"Error serializing matching results to JSON: {e}")
            raise


class XMLSerializer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.namespace_map = {
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'owl': 'http://www.w3.org/2002/07/owl#',
            'tmf': 'http://www.tmforum.org/ontology#'
        }
    
    def serialize_ontology(self, ontology: TMFOntology, file_path: str) -> None:
        try:
            root = self._create_rdf_root(ontology)
            
            for entity in ontology.entities.values():
                entity_element = self._create_entity_element(entity)
                root.append(entity_element)
            
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ", level=0)
            tree.write(file_path, encoding='utf-8', xml_declaration=True)
            
            self.logger.info(f"Ontology serialized to XML: {file_path}")
        except Exception as e:
            self.logger.error(f"Error serializing ontology to XML: {e}")
            raise
    
    def _create_rdf_root(self, ontology: TMFOntology) -> ET.Element:
        root = ET.Element(f"{{{self.namespace_map['rdf']}}}RDF")
        
        for prefix, uri in self.namespace_map.items():
            root.set(f"xmlns:{prefix}", uri)
        
        ontology_element = ET.SubElement(root, f"{{{self.namespace_map['owl']}}}Ontology")
        ontology_element.set(f"{{{self.namespace_map['rdf']}}}about", ontology.namespace)
        
        version_element = ET.SubElement(ontology_element, f"{{{self.namespace_map['owl']}}}versionInfo")
        version_element.text = ontology.version
        
        return root
    
    def _create_entity_element(self, entity: TMFEntity) -> ET.Element:
        element_type = self._map_entity_type_to_owl(entity.entity_type)
        element = ET.Element(f"{{{self.namespace_map['owl']}}}{element_type}")
        element.set(f"{{{self.namespace_map['rdf']}}}about", entity.id)
        
        if entity.name:
            label_element = ET.SubElement(element, f"{{{self.namespace_map['rdfs']}}}label")
            label_element.text = entity.name
        
        if entity.description:
            comment_element = ET.SubElement(element, f"{{{self.namespace_map['rdfs']}}}comment")
            comment_element.text = entity.description
        
        if entity.parent_id:
            subclass_element = ET.SubElement(element, f"{{{self.namespace_map['rdfs']}}}subClassOf")
            subclass_element.set(f"{{{self.namespace_map['rdf']}}}resource", entity.parent_id)
        
        for relationship_id in entity.relationships:
            related_element = ET.SubElement(element, f"{{{self.namespace_map['rdfs']}}}seeAlso")
            related_element.set(f"{{{self.namespace_map['rdf']}}}resource", relationship_id)
        
        for key, value in entity.properties.items():
            prop_element = ET.SubElement(element, f"{{{self.namespace_map['tmf']}}}{key}")
            prop_element.text = str(value)
        
        return element
    
    def _map_entity_type_to_owl(self, entity_type) -> str:
        mapping = {
            'concept': 'Class',
            'business_process': 'Class',
            'information_model': 'Class',
            'functional_model': 'Class',
            'component_model': 'Class',
            'property': 'DatatypeProperty',
            'relationship': 'ObjectProperty'
        }
        return mapping.get(entity_type.value, 'Class')
    
    def serialize_matching_results(self, results: Dict[str, Any], file_path: str) -> None:
        try:
            root = ET.Element("MatchingResults")
            
            metadata_element = ET.SubElement(root, "Metadata")
            for key, value in results.get('metadata', {}).items():
                meta_element = ET.SubElement(metadata_element, key)
                meta_element.text = str(value)
            
            matches_element = ET.SubElement(root, "Matches")
            for match in results.get('matches', []):
                match_element = ET.SubElement(matches_element, "Match")
                
                for key, value in match.items():
                    if isinstance(value, (list, dict)):
                        continue
                    element = ET.SubElement(match_element, key)
                    element.text = str(value)
            
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ", level=0)
            tree.write(file_path, encoding='utf-8', xml_declaration=True)
            
            self.logger.info(f"Matching results serialized to XML: {file_path}")
        except Exception as e:
            self.logger.error(f"Error serializing matching results to XML: {e}")
            raise


class UniversalSerializer:
    def __init__(self):
        self.json_serializer = JSONSerializer()
        self.xml_serializer = XMLSerializer()
        self.logger = logging.getLogger(__name__)
    
    def serialize_ontology(self, ontology: TMFOntology, file_path: str) -> None:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.json':
            self.json_serializer.serialize_ontology(ontology, file_path)
        elif file_extension in ['.xml', '.owl', '.rdf']:
            self.xml_serializer.serialize_ontology(ontology, file_path)
        else:
            self.logger.warning(f"Unknown format {file_extension}, defaulting to JSON")
            json_path = str(Path(file_path).with_suffix('.json'))
            self.json_serializer.serialize_ontology(ontology, json_path)
    
    def deserialize_ontology(self, file_path: str) -> TMFOntology:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.json':
            return self.json_serializer.deserialize_ontology(file_path)
        else:
            raise ValueError(f"Deserialization not supported for format: {file_extension}")
    
    def serialize_matching_results(self, results: Dict[str, Any], file_path: str) -> None:
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.json':
            self.json_serializer.serialize_matching_results(results, file_path)
        elif file_extension in ['.xml']:
            self.xml_serializer.serialize_matching_results(results, file_path)
        else:
            self.logger.warning(f"Unknown format {file_extension}, defaulting to JSON")
            json_path = str(Path(file_path).with_suffix('.json'))
            self.json_serializer.serialize_matching_results(results, json_path)
