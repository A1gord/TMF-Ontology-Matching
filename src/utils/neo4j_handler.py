from typing import Dict, List, Any, Optional, Tuple
import logging
from dataclasses import dataclass
import json
from neo4j import GraphDatabase, Driver, Session
from neo4j.exceptions import ServiceUnavailable, AuthError

from ..entities.tmf_entity import TMFEntity, TMFOntology, EntityType
from ..entities.knowledge_graph import TMFKnowledgeGraph
from ..matching.matching_engine import MatchingResult
from ..alignment.alignment_engine import AlignmentResult


class Neo4jConnector:
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.driver: Optional[Driver] = None
        self._connect()

    def _connect(self):
        try:
            self.driver = GraphDatabase.driver(
                self.config.uri,
                auth=(self.config.username, self.config.password),
                max_connection_lifetime=self.config.max_connection_lifetime,
                max_connection_pool_size=self.config.max_connection_pool_size,
                connection_acquisition_timeout=self.config.connection_acquisition_timeout,
            )
            self.driver.verify_connectivity()
            self.logger.info(f"Connected to Neo4j at {self.config.uri}")
        except (ServiceUnavailable, AuthError) as e:
            self.logger.error(f"Failed to connect to Neo4j: {e}")
            raise

    def close(self):
        if self.driver:
            self.driver.close()
            self.logger.info("Neo4j connection closed")

    def get_session(self) -> Session:
        if not self.driver:
            raise RuntimeError("Neo4j driver not initialized")
        return self.driver.session(database=self.config.database)

    def execute_query(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def execute_write_query(
        self, query: str, parameters: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        with self.get_session() as session:
            result = session.write_transaction(
                lambda tx: tx.run(query, parameters or {})
            )
            return [record.data() for record in result]


class TMFNeo4jExporter:
    def __init__(self, connector: Neo4jConnector):
        self.connector = connector
        self.logger = logging.getLogger(__name__)

    def export_ontology(self, ontology: TMFOntology) -> None:
        try:
            self._clear_ontology_data(ontology.name)
            self._create_entities(ontology)
            self._create_relationships(ontology)
            self.logger.info(f"Ontology '{ontology.name}' exported to Neo4j")
        except Exception as e:
            self.logger.error(f"Error exporting ontology to Neo4j: {e}")
            raise

    def _clear_ontology_data(self, ontology_name: str) -> None:
        query = """
        MATCH (n {ontology_name: $ontology_name})
        DETACH DELETE n
        """
        self.connector.execute_write_query(query, {"ontology_name": ontology_name})

    def _create_entities(self, ontology: TMFOntology) -> None:
        for entity in ontology.entities.values():
            self._create_entity_node(entity, ontology.name)

    def _create_entity_node(self, entity: TMFEntity, ontology_name: str) -> None:
        query = """
        CREATE (e:TMFEntity {
            id: $id,
            name: $name,
            description: $description,
            entity_type: $entity_type,
            namespace: $namespace,
            source_ontology: $source_ontology,
            ontology_name: $ontology_name,
            confidence_score: $confidence_score,
            properties: $properties,
            synonyms: $synonyms
        })
        """

        parameters = {
            "id": entity.id,
            "name": entity.name,
            "description": entity.description,
            "entity_type": entity.entity_type.value,
            "namespace": entity.namespace,
            "source_ontology": entity.source_ontology,
            "ontology_name": ontology_name,
            "confidence_score": entity.confidence_score,
            "properties": json.dumps(entity.properties),
            "synonyms": entity.synonyms,
        }

        self.connector.execute_write_query(query, parameters)

    def _create_relationships(self, ontology: TMFOntology) -> None:
        for entity in ontology.entities.values():
            if entity.parent_id:
                self._create_hierarchy_relationship(
                    entity.id, entity.parent_id, ontology.name
                )

            for child_id in entity.children_ids:
                self._create_hierarchy_relationship(entity.id, child_id, ontology.name)

            for related_id in entity.relationships:
                self._create_semantic_relationship(entity.id, related_id, ontology.name)

    def _create_hierarchy_relationship(
        self, source_id: str, target_id: str, ontology_name: str
    ) -> None:
        query = """
        MATCH (source:TMFEntity {id: $source_id, ontology_name: $ontology_name})
        MATCH (target:TMFEntity {id: $target_id, ontology_name: $ontology_name})
        CREATE (source)-[:PARENT_OF]->(target)
        """

        parameters = {
            "source_id": source_id,
            "target_id": target_id,
            "ontology_name": ontology_name,
        }

        self.connector.execute_write_query(query, parameters)

    def _create_semantic_relationship(
        self, source_id: str, target_id: str, ontology_name: str
    ) -> None:
        query = """
        MATCH (source:TMFEntity {id: $source_id, ontology_name: $ontology_name})
        MATCH (target:TMFEntity {id: $target_id, ontology_name: $ontology_name})
        CREATE (source)-[:RELATED_TO]->(target)
        """

        parameters = {
            "source_id": source_id,
            "target_id": target_id,
            "ontology_name": ontology_name,
        }

        self.connector.execute_write_query(query, parameters)

    def export_matching_results(
        self, results: List[MatchingResult], ontology1_name: str, ontology2_name: str
    ) -> None:
        try:
            self._clear_matching_results(ontology1_name, ontology2_name)

            for result in results:
                self._create_match_relationship(result, ontology1_name, ontology2_name)

            self.logger.info(
                f"Matching results exported to Neo4j for {ontology1_name} -> {ontology2_name}"
            )
        except Exception as e:
            self.logger.error(f"Error exporting matching results to Neo4j: {e}")
            raise

    def _clear_matching_results(self, ontology1_name: str, ontology2_name: str) -> None:
        query = """
        MATCH (source:TMFEntity {ontology_name: $ontology1_name})-[r:MATCHES]->(target:TMFEntity {ontology_name: $ontology2_name})
        DELETE r
        """

        parameters = {
            "ontology1_name": ontology1_name,
            "ontology2_name": ontology2_name,
        }

        self.connector.execute_write_query(query, parameters)

    def _create_match_relationship(
        self, result: MatchingResult, ontology1_name: str, ontology2_name: str
    ) -> None:
        query = """
        MATCH (source:TMFEntity {id: $source_id, ontology_name: $ontology1_name})
        MATCH (target:TMFEntity {id: $target_id, ontology_name: $ontology2_name})
        CREATE (source)-[:MATCHES {
            similarity_score: $similarity_score,
            confidence: $confidence,
            match_type: $match_type,
            details: $details,
            timestamp: $timestamp
        }]->(target)
        """

        parameters = {
            "source_id": result.source_entity.id,
            "target_id": result.target_entity.id,
            "ontology1_name": ontology1_name,
            "ontology2_name": ontology2_name,
            "similarity_score": result.similarity_score,
            "confidence": result.confidence,
            "match_type": result.match_type,
            "details": json.dumps(result.details),
            "timestamp": result.timestamp,
        }

        self.connector.execute_write_query(query, parameters)

    def export_alignment_results(
        self, results: List[AlignmentResult], ontology1_name: str, ontology2_name: str
    ) -> None:
        try:
            self._clear_alignment_results(ontology1_name, ontology2_name)

            for result in results:
                self._create_alignment_relationship(
                    result, ontology1_name, ontology2_name
                )

            self.logger.info(
                f"Alignment results exported to Neo4j for {ontology1_name} -> {ontology2_name}"
            )
        except Exception as e:
            self.logger.error(f"Error exporting alignment results to Neo4j: {e}")
            raise

    def _clear_alignment_results(
        self, ontology1_name: str, ontology2_name: str
    ) -> None:
        query = """
        MATCH (source:TMFEntity {ontology_name: $ontology1_name})-[r:ALIGNED_WITH]->(target:TMFEntity {ontology_name: $ontology2_name})
        DELETE r
        """

        parameters = {
            "ontology1_name": ontology1_name,
            "ontology2_name": ontology2_name,
        }

        self.connector.execute_write_query(query, parameters)

    def _create_alignment_relationship(
        self, result: AlignmentResult, ontology1_name: str, ontology2_name: str
    ) -> None:
        query = """
        MATCH (source:TMFEntity {id: $source_id, ontology_name: $ontology1_name})
        MATCH (target:TMFEntity {id: $target_id, ontology_name: $ontology2_name})
        CREATE (source)-[:ALIGNED_WITH {
            alignment_type: $alignment_type,
            confidence: $confidence,
            hierarchical_distance: $hierarchical_distance,
            structural_similarity: $structural_similarity,
            details: $details,
            timestamp: $timestamp
        }]->(target)
        """

        parameters = {
            "source_id": result.source_entity.id,
            "target_id": result.target_entity.id,
            "ontology1_name": ontology1_name,
            "ontology2_name": ontology2_name,
            "alignment_type": result.alignment_type.value,
            "confidence": result.confidence,
            "hierarchical_distance": result.hierarchical_distance,
            "structural_similarity": result.structural_similarity,
            "details": json.dumps(result.details),
            "timestamp": result.timestamp,
        }

        self.connector.execute_write_query(query, parameters)


class TMFNeo4jImporter:
    def __init__(self, connector: Neo4jConnector):
        self.connector = connector
        self.logger = logging.getLogger(__name__)

    def import_ontology(self, ontology_name: str) -> TMFOntology:
        try:
            ontology = TMFOntology(name=ontology_name)

            entities = self._load_entities(ontology_name)
            for entity in entities:
                ontology.add_entity(entity)

            self.logger.info(f"Ontology '{ontology_name}' imported from Neo4j")
            return ontology

        except Exception as e:
            self.logger.error(f"Error importing ontology from Neo4j: {e}")
            raise

    def _load_entities(self, ontology_name: str) -> List[TMFEntity]:
        query = """
        MATCH (e:TMFEntity {ontology_name: $ontology_name})
        RETURN e
        """

        results = self.connector.execute_query(query, {"ontology_name": ontology_name})
        entities = []

        for record in results:
            entity_data = record["e"]
            entity = self._create_entity_from_node(entity_data)
            entities.append(entity)

        self._load_relationships(entities, ontology_name)
        return entities

    def _create_entity_from_node(self, node_data: Dict[str, Any]) -> TMFEntity:
        entity = TMFEntity(
            id=node_data["id"],
            name=node_data["name"],
            description=node_data.get("description", ""),
            entity_type=EntityType(node_data["entity_type"]),
            namespace=node_data.get("namespace", ""),
            source_ontology=node_data.get("source_ontology", ""),
            confidence_score=node_data.get("confidence_score", 1.0),
        )

        if "properties" in node_data and node_data["properties"]:
            try:
                entity.properties = json.loads(node_data["properties"])
            except json.JSONDecodeError:
                pass

        if "synonyms" in node_data and node_data["synonyms"]:
            entity.synonyms = node_data["synonyms"]

        return entity

    def _load_relationships(
        self, entities: List[TMFEntity], ontology_name: str
    ) -> None:
        entity_map = {entity.id: entity for entity in entities}

        hierarchy_query = """
        MATCH (source:TMFEntity {ontology_name: $ontology_name})-[:PARENT_OF]->(target:TMFEntity {ontology_name: $ontology_name})
        RETURN source.id as source_id, target.id as target_id
        """

        hierarchy_results = self.connector.execute_query(
            hierarchy_query, {"ontology_name": ontology_name}
        )

        for record in hierarchy_results:
            source_id = record["source_id"]
            target_id = record["target_id"]

            if source_id in entity_map and target_id in entity_map:
                entity_map[source_id].add_child(target_id)
                entity_map[target_id].parent_id = source_id

        semantic_query = """
        MATCH (source:TMFEntity {ontology_name: $ontology_name})-[:RELATED_TO]->(target:TMFEntity {ontology_name: $ontology_name})
        RETURN source.id as source_id, target.id as target_id
        """

        semantic_results = self.connector.execute_query(
            semantic_query, {"ontology_name": ontology_name}
        )

        for record in semantic_results:
            source_id = record["source_id"]
            target_id = record["target_id"]

            if source_id in entity_map:
                entity_map[source_id].add_relationship(target_id)

    def import_matching_results(
        self, ontology1_name: str, ontology2_name: str
    ) -> List[MatchingResult]:
        try:
            query = """
            MATCH (source:TMFEntity {ontology_name: $ontology1_name})-[r:MATCHES]->(target:TMFEntity {ontology_name: $ontology2_name})
            RETURN source, target, r
            """

            parameters = {
                "ontology1_name": ontology1_name,
                "ontology2_name": ontology2_name,
            }

            results = self.connector.execute_query(query, parameters)
            matching_results = []

            for record in results:
                source_entity = self._create_entity_from_node(record["source"])
                target_entity = self._create_entity_from_node(record["target"])
                relationship = record["r"]

                details = {}
                if "details" in relationship and relationship["details"]:
                    try:
                        details = json.loads(relationship["details"])
                    except json.JSONDecodeError:
                        pass

                match_result = MatchingResult(
                    source_entity=source_entity,
                    target_entity=target_entity,
                    similarity_score=relationship["similarity_score"],
                    confidence=relationship["confidence"],
                    match_type=relationship["match_type"],
                    details=details,
                    timestamp=relationship.get("timestamp", 0.0),
                )

                matching_results.append(match_result)

            self.logger.info(
                f"Imported {len(matching_results)} matching results from Neo4j"
            )
            return matching_results

        except Exception as e:
            self.logger.error(f"Error importing matching results from Neo4j: {e}")
            raise

    def import_alignment_results(
        self, ontology1_name: str, ontology2_name: str
    ) -> List[AlignmentResult]:
        try:
            from ..alignment.alignment_engine import AlignmentType

            query = """
            MATCH (source:TMFEntity {ontology_name: $ontology1_name})-[r:ALIGNED_WITH]->(target:TMFEntity {ontology_name: $ontology2_name})
            RETURN source, target, r
            """

            parameters = {
                "ontology1_name": ontology1_name,
                "ontology2_name": ontology2_name,
            }

            results = self.connector.execute_query(query, parameters)
            alignment_results = []

            for record in results:
                source_entity = self._create_entity_from_node(record["source"])
                target_entity = self._create_entity_from_node(record["target"])
                relationship = record["r"]

                details = {}
                if "details" in relationship and relationship["details"]:
                    try:
                        details = json.loads(relationship["details"])
                    except json.JSONDecodeError:
                        pass

                alignment_result = AlignmentResult(
                    source_entity=source_entity,
                    target_entity=target_entity,
                    alignment_type=AlignmentType(relationship["alignment_type"]),
                    confidence=relationship["confidence"],
                    hierarchical_distance=relationship["hierarchical_distance"],
                    structural_similarity=relationship["structural_similarity"],
                    details=details,
                    timestamp=relationship.get("timestamp", 0.0),
                )

                alignment_results.append(alignment_result)

            self.logger.info(
                f"Imported {len(alignment_results)} alignment results from Neo4j"
            )
            return alignment_results

        except Exception as e:
            self.logger.error(f"Error importing alignment results from Neo4j: {e}")
            raise

    def get_ontology_statistics(self, ontology_name: str) -> Dict[str, Any]:
        try:
            stats_query = """
            MATCH (e:TMFEntity {ontology_name: $ontology_name})
            RETURN 
                count(e) as total_entities,
                collect(DISTINCT e.entity_type) as entity_types
            """

            relationships_query = """
            MATCH (source:TMFEntity {ontology_name: $ontology_name})-[r]->(target:TMFEntity {ontology_name: $ontology_name})
            RETURN type(r) as relationship_type, count(r) as count
            """

            stats_result = self.connector.execute_query(
                stats_query, {"ontology_name": ontology_name}
            )
            relationships_result = self.connector.execute_query(
                relationships_query, {"ontology_name": ontology_name}
            )

            statistics = {
                "ontology_name": ontology_name,
                "total_entities": (
                    stats_result[0]["total_entities"] if stats_result else 0
                ),
                "entity_types": stats_result[0]["entity_types"] if stats_result else [],
                "relationships": {
                    r["relationship_type"]: r["count"] for r in relationships_result
                },
            }

            return statistics

        except Exception as e:
            self.logger.error(f"Error getting ontology statistics from Neo4j: {e}")
            raise
