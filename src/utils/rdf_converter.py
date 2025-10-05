from typing import Dict, List, Any, Optional
import logging
from rdflib import Graph, Namespace, URIRef, Literal, BNode
from rdflib.namespace import RDF, RDFS, OWL, XSD
import json
from pathlib import Path

from ..entities.tmf_entity import TMFEntity, TMFOntology, EntityType
from ..entities.knowledge_graph import TMFKnowledgeGraph
from ..matching.matching_engine import MatchingResult
from ..alignment.alignment_engine import AlignmentResult
from .neo4j_handler import Neo4jConnector


class RDFConverter:
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        default_namespace = "http://www.tmforum.org/ontology#"
        if config and hasattr(config, 'parsing') and config.parsing.default_namespace:
            default_namespace = config.parsing.default_namespace
        
        self.tmf_ns = Namespace(default_namespace)
        self.matching_ns = Namespace("http://www.tmforum.org/matching#")
        self.alignment_ns = Namespace("http://www.tmforum.org/alignment#")
    
    def ontology_to_rdf(self, ontology: TMFOntology) -> Graph:
        try:
            graph = Graph()
            
            graph.bind("tmf", self.tmf_ns)
            graph.bind("rdf", RDF)
            graph.bind("rdfs", RDFS)
            graph.bind("owl", OWL)
            
            default_base_uri = "http://example.org"
            if self.config and hasattr(self.config, 'parsing'):
                default_base_uri = self.config.parsing.default_namespace.rstrip('#/')
            
            ontology_uri = URIRef(ontology.namespace or f"{default_base_uri}/{ontology.name}")
            graph.add((ontology_uri, RDF.type, OWL.Ontology))
            graph.add((ontology_uri, RDFS.label, Literal(ontology.name)))
            graph.add((ontology_uri, OWL.versionInfo, Literal(ontology.version)))
            
            for entity in ontology.entities.values():
                self._add_entity_to_graph(graph, entity, ontology_uri)
            
            self.logger.info(f"Converted ontology '{ontology.name}' to RDF with {len(graph)} triples")
            return graph
            
        except Exception as e:
            self.logger.error(f"Error converting ontology to RDF: {e}")
            raise
    
    def _add_entity_to_graph(self, graph: Graph, entity: TMFEntity, ontology_uri: URIRef) -> None:
        entity_uri = URIRef(f"{ontology_uri}#{entity.id}")
        
        entity_class = self._get_owl_class_for_entity_type(entity.entity_type)
        graph.add((entity_uri, RDF.type, entity_class))
        
        if entity.name:
            graph.add((entity_uri, RDFS.label, Literal(entity.name)))
        
        if entity.description:
            graph.add((entity_uri, RDFS.comment, Literal(entity.description)))
        
        if entity.parent_id:
            parent_uri = URIRef(f"{ontology_uri}#{entity.parent_id}")
            graph.add((entity_uri, RDFS.subClassOf, parent_uri))
        
        for synonym in entity.synonyms:
            graph.add((entity_uri, self.tmf_ns.synonym, Literal(synonym)))
        
        for child_id in entity.children_ids:
            child_uri = URIRef(f"{ontology_uri}#{child_id}")
            graph.add((child_uri, RDFS.subClassOf, entity_uri))
        
        for related_id in entity.relationships:
            related_uri = URIRef(f"{ontology_uri}#{related_id}")
            graph.add((entity_uri, RDFS.seeAlso, related_uri))
        
        for key, value in entity.properties.items():
            property_uri = self.tmf_ns[key]
            if isinstance(value, (int, float)):
                graph.add((entity_uri, property_uri, Literal(value, datatype=XSD.decimal)))
            elif isinstance(value, bool):
                graph.add((entity_uri, property_uri, Literal(value, datatype=XSD.boolean)))
            else:
                graph.add((entity_uri, property_uri, Literal(str(value))))
        
        graph.add((entity_uri, self.tmf_ns.entityType, Literal(entity.entity_type.value)))
        graph.add((entity_uri, self.tmf_ns.confidenceScore, Literal(entity.confidence_score, datatype=XSD.decimal)))
        graph.add((entity_uri, self.tmf_ns.sourceOntology, Literal(entity.source_ontology)))
    
    def _get_owl_class_for_entity_type(self, entity_type: EntityType) -> URIRef:
        mapping = {
            EntityType.CONCEPT: OWL.Class,
            EntityType.BUSINESS_PROCESS: OWL.Class,
            EntityType.INFORMATION_MODEL: OWL.Class,
            EntityType.FUNCTIONAL_MODEL: OWL.Class,
            EntityType.COMPONENT_MODEL: OWL.Class,
            EntityType.PROPERTY: OWL.DatatypeProperty,
            EntityType.RELATIONSHIP: OWL.ObjectProperty
        }
        return mapping.get(entity_type, OWL.Class)
    
    def rdf_to_ontology(self, graph: Graph, ontology_name: str = None) -> TMFOntology:
        try:
            ontology_uris = list(graph.subjects(RDF.type, OWL.Ontology))
            
            if ontology_uris:
                ontology_uri = ontology_uris[0]
                name = str(graph.value(ontology_uri, RDFS.label)) or ontology_name or "Imported_Ontology"
                version = str(graph.value(ontology_uri, OWL.versionInfo)) or "1.0"
                namespace = str(ontology_uri)
            else:
                name = ontology_name or "Imported_Ontology"
                version = "1.0"
                default_base_uri = "http://example.org"
                if self.config and hasattr(self.config, 'parsing'):
                    default_base_uri = self.config.parsing.default_namespace.rstrip('#/')
                namespace = f"{default_base_uri}/imported"
            
            ontology = TMFOntology(name=name, version=version, namespace=namespace)
            
            classes = list(graph.subjects(RDF.type, OWL.Class))
            properties = list(graph.subjects(RDF.type, OWL.DatatypeProperty)) + \
                        list(graph.subjects(RDF.type, OWL.ObjectProperty))
            
            all_entities = classes + properties
            
            for entity_uri in all_entities:
                entity = self._create_entity_from_rdf(graph, entity_uri, namespace)
                if entity:
                    ontology.add_entity(entity)
            
            self.logger.info(f"Converted RDF to ontology '{name}' with {len(ontology.entities)} entities")
            return ontology
            
        except Exception as e:
            self.logger.error(f"Error converting RDF to ontology: {e}")
            raise
    
    def _create_entity_from_rdf(self, graph: Graph, entity_uri: URIRef, base_namespace: str) -> Optional[TMFEntity]:
        try:
            entity_id = str(entity_uri).replace(f"{base_namespace}#", "")
            
            name = str(graph.value(entity_uri, RDFS.label)) or entity_id
            description = str(graph.value(entity_uri, RDFS.comment)) or ""
            
            entity_type_value = graph.value(entity_uri, self.tmf_ns.entityType)
            if entity_type_value:
                entity_type = EntityType(str(entity_type_value))
            else:
                entity_type = self._infer_entity_type_from_rdf_type(graph, entity_uri)
            
            entity = TMFEntity(
                id=entity_id,
                name=name,
                description=description,
                entity_type=entity_type,
                namespace=base_namespace
            )
            
            confidence_score = graph.value(entity_uri, self.tmf_ns.confidenceScore)
            if confidence_score:
                entity.confidence_score = float(confidence_score)
            
            source_ontology = graph.value(entity_uri, self.tmf_ns.sourceOntology)
            if source_ontology:
                entity.source_ontology = str(source_ontology)
            
            synonyms = list(graph.objects(entity_uri, self.tmf_ns.synonym))
            for synonym in synonyms:
                entity.add_synonym(str(synonym))
            
            parent_classes = list(graph.objects(entity_uri, RDFS.subClassOf))
            for parent_class in parent_classes:
                if isinstance(parent_class, URIRef):
                    parent_id = str(parent_class).replace(f"{base_namespace}#", "")
                    entity.parent_id = parent_id
                    break
            
            subclasses = list(graph.subjects(RDFS.subClassOf, entity_uri))
            for subclass in subclasses:
                if isinstance(subclass, URIRef):
                    child_id = str(subclass).replace(f"{base_namespace}#", "")
                    entity.add_child(child_id)
            
            related_entities = list(graph.objects(entity_uri, RDFS.seeAlso))
            for related_entity in related_entities:
                if isinstance(related_entity, URIRef):
                    related_id = str(related_entity).replace(f"{base_namespace}#", "")
                    entity.add_relationship(related_id)
            
            for predicate, obj in graph.predicate_objects(entity_uri):
                if predicate.startswith(self.tmf_ns) and predicate not in [
                    self.tmf_ns.synonym, self.tmf_ns.entityType, 
                    self.tmf_ns.confidenceScore, self.tmf_ns.sourceOntology
                ]:
                    property_name = str(predicate).replace(str(self.tmf_ns), "")
                    entity.add_property(property_name, str(obj))
            
            return entity
            
        except Exception as e:
            self.logger.error(f"Error creating entity from RDF: {e}")
            return None
    
    def _infer_entity_type_from_rdf_type(self, graph: Graph, entity_uri: URIRef) -> EntityType:
        rdf_types = list(graph.objects(entity_uri, RDF.type))
        
        for rdf_type in rdf_types:
            if rdf_type == OWL.Class:
                return EntityType.CONCEPT
            elif rdf_type == OWL.DatatypeProperty:
                return EntityType.PROPERTY
            elif rdf_type == OWL.ObjectProperty:
                return EntityType.RELATIONSHIP
        
        return EntityType.CONCEPT
    
    def matching_results_to_rdf(self, results: List[MatchingResult], 
                               source_ontology_uri: str, target_ontology_uri: str) -> Graph:
        try:
            graph = Graph()
            
            graph.bind("matching", self.matching_ns)
            graph.bind("rdf", RDF)
            graph.bind("rdfs", RDFS)
            graph.bind("owl", OWL)
            
            for i, result in enumerate(results):
                match_uri = URIRef(f"{self.matching_ns}match_{i}")
                
                graph.add((match_uri, RDF.type, self.matching_ns.Match))
                
                source_uri = URIRef(f"{source_ontology_uri}#{result.source_entity.id}")
                target_uri = URIRef(f"{target_ontology_uri}#{result.target_entity.id}")
                
                graph.add((match_uri, self.matching_ns.sourceEntity, source_uri))
                graph.add((match_uri, self.matching_ns.targetEntity, target_uri))
                graph.add((match_uri, self.matching_ns.similarityScore, 
                          Literal(result.similarity_score, datatype=XSD.decimal)))
                graph.add((match_uri, self.matching_ns.confidence, 
                          Literal(result.confidence, datatype=XSD.decimal)))
                graph.add((match_uri, self.matching_ns.matchType, Literal(result.match_type)))
                graph.add((match_uri, self.matching_ns.timestamp, 
                          Literal(result.timestamp, datatype=XSD.decimal)))
                
                if result.details:
                    details_json = json.dumps(result.details)
                    graph.add((match_uri, self.matching_ns.details, Literal(details_json)))
            
            self.logger.info(f"Converted {len(results)} matching results to RDF")
            return graph
            
        except Exception as e:
            self.logger.error(f"Error converting matching results to RDF: {e}")
            raise
    
    def alignment_results_to_rdf(self, results: List[AlignmentResult], 
                                source_ontology_uri: str, target_ontology_uri: str) -> Graph:
        try:
            graph = Graph()
            
            graph.bind("alignment", self.alignment_ns)
            graph.bind("rdf", RDF)
            graph.bind("rdfs", RDFS)
            graph.bind("owl", OWL)
            
            for i, result in enumerate(results):
                alignment_uri = URIRef(f"{self.alignment_ns}alignment_{i}")
                
                graph.add((alignment_uri, RDF.type, self.alignment_ns.Alignment))
                
                source_uri = URIRef(f"{source_ontology_uri}#{result.source_entity.id}")
                target_uri = URIRef(f"{target_ontology_uri}#{result.target_entity.id}")
                
                graph.add((alignment_uri, self.alignment_ns.sourceEntity, source_uri))
                graph.add((alignment_uri, self.alignment_ns.targetEntity, target_uri))
                graph.add((alignment_uri, self.alignment_ns.alignmentType, 
                          Literal(result.alignment_type.value)))
                graph.add((alignment_uri, self.alignment_ns.confidence, 
                          Literal(result.confidence, datatype=XSD.decimal)))
                graph.add((alignment_uri, self.alignment_ns.hierarchicalDistance, 
                          Literal(result.hierarchical_distance, datatype=XSD.decimal)))
                graph.add((alignment_uri, self.alignment_ns.structuralSimilarity, 
                          Literal(result.structural_similarity, datatype=XSD.decimal)))
                graph.add((alignment_uri, self.alignment_ns.timestamp, 
                          Literal(result.timestamp, datatype=XSD.decimal)))
                
                if result.details:
                    details_json = json.dumps(result.details)
                    graph.add((alignment_uri, self.alignment_ns.details, Literal(details_json)))
            
            self.logger.info(f"Converted {len(results)} alignment results to RDF")
            return graph
            
        except Exception as e:
            self.logger.error(f"Error converting alignment results to RDF: {e}")
            raise
    
    def save_rdf_to_file(self, graph: Graph, file_path: str, format: str = None) -> None:
        try:
            if format is None:
                format = "turtle"
                if self.config and hasattr(self.config, 'serialization'):
                    format_mapping = {
                        "json": "json-ld",
                        "xml": "xml", 
                        "turtle": "turtle"
                    }
                    format = format_mapping.get(self.config.serialization.default_format, "turtle")
            
            graph.serialize(destination=file_path, format=format)
            self.logger.info(f"RDF graph saved to {file_path} in {format} format")
        except Exception as e:
            self.logger.error(f"Error saving RDF to file: {e}")
            raise
    
    def load_rdf_from_file(self, file_path: str) -> Graph:
        try:
            graph = Graph()
            graph.parse(file_path)
            self.logger.info(f"RDF graph loaded from {file_path}")
            return graph
        except Exception as e:
            self.logger.error(f"Error loading RDF from file: {e}")
            raise


class Neo4jToRDFConverter:
    def __init__(self, neo4j_connector: Neo4jConnector):
        self.neo4j_connector = neo4j_connector
        self.rdf_converter = RDFConverter()
        self.logger = logging.getLogger(__name__)
    
    def export_ontology_from_neo4j_to_rdf(self, ontology_name: str, output_path: str, 
                                         format: str = "turtle") -> None:
        try:
            from .neo4j_handler import TMFNeo4jImporter
            
            importer = TMFNeo4jImporter(self.neo4j_connector)
            ontology = importer.import_ontology(ontology_name)
            
            rdf_graph = self.rdf_converter.ontology_to_rdf(ontology)
            
            self.rdf_converter.save_rdf_to_file(rdf_graph, output_path, format)
            
            self.logger.info(f"Exported ontology '{ontology_name}' from Neo4j to RDF file '{output_path}'")
            
        except Exception as e:
            self.logger.error(f"Error exporting ontology from Neo4j to RDF: {e}")
            raise
    
    def export_matching_results_from_neo4j_to_rdf(self, ontology1_name: str, ontology2_name: str,
                                                 output_path: str, format: str = "turtle") -> None:
        try:
            from .neo4j_handler import TMFNeo4jImporter
            
            importer = TMFNeo4jImporter(self.neo4j_connector)
            matching_results = importer.import_matching_results(ontology1_name, ontology2_name)
            
            source_uri = f"http://example.org/{ontology1_name}"
            target_uri = f"http://example.org/{ontology2_name}"
            
            rdf_graph = self.rdf_converter.matching_results_to_rdf(
                matching_results, source_uri, target_uri)
            
            self.rdf_converter.save_rdf_to_file(rdf_graph, output_path, format)
            
            self.logger.info(f"Exported matching results from Neo4j to RDF file '{output_path}'")
            
        except Exception as e:
            self.logger.error(f"Error exporting matching results from Neo4j to RDF: {e}")
            raise
    
    def export_alignment_results_from_neo4j_to_rdf(self, ontology1_name: str, ontology2_name: str,
                                                  output_path: str, format: str = "turtle") -> None:
        try:
            from .neo4j_handler import TMFNeo4jImporter
            
            importer = TMFNeo4jImporter(self.neo4j_connector)
            alignment_results = importer.import_alignment_results(ontology1_name, ontology2_name)
            
            source_uri = f"http://example.org/{ontology1_name}"
            target_uri = f"http://example.org/{ontology2_name}"
            
            rdf_graph = self.rdf_converter.alignment_results_to_rdf(
                alignment_results, source_uri, target_uri)
            
            self.rdf_converter.save_rdf_to_file(rdf_graph, output_path, format)
            
            self.logger.info(f"Exported alignment results from Neo4j to RDF file '{output_path}'")
            
        except Exception as e:
            self.logger.error(f"Error exporting alignment results from Neo4j to RDF: {e}")
            raise
    
    def import_rdf_to_neo4j(self, rdf_file_path: str, ontology_name: str) -> None:
        try:
            from .neo4j_handler import TMFNeo4jExporter
            
            rdf_graph = self.rdf_converter.load_rdf_from_file(rdf_file_path)
            ontology = self.rdf_converter.rdf_to_ontology(rdf_graph, ontology_name)
            
            exporter = TMFNeo4jExporter(self.neo4j_connector)
            exporter.export_ontology(ontology)
            
            self.logger.info(f"Imported RDF file '{rdf_file_path}' to Neo4j as ontology '{ontology_name}'")
            
        except Exception as e:
            self.logger.error(f"Error importing RDF to Neo4j: {e}")
            raise
