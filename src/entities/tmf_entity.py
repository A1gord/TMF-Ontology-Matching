from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
import uuid


class EntityType(Enum):
    BUSINESS_PROCESS = "business_process"
    INFORMATION_MODEL = "information_model"
    FUNCTIONAL_MODEL = "functional_model"
    COMPONENT_MODEL = "component_model"
    CONCEPT = "concept"
    PROPERTY = "property"
    RELATIONSHIP = "relationship"


@dataclass
class TMFEntity:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    entity_type: EntityType = EntityType.CONCEPT
    properties: Dict[str, Any] = field(default_factory=dict)
    relationships: List[str] = field(default_factory=list)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    namespace: str = ""
    source_ontology: str = ""
    confidence_score: float = 1.0

    def add_property(self, key: str, value: Any) -> None:
        self.properties[key] = value

    def add_relationship(self, entity_id: str) -> None:
        if entity_id not in self.relationships:
            self.relationships.append(entity_id)

    def add_child(self, child_id: str) -> None:
        if child_id not in self.children_ids:
            self.children_ids.append(child_id)

    def add_synonym(self, synonym: str) -> None:
        if synonym not in self.synonyms:
            self.synonyms.append(synonym)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "entity_type": self.entity_type.value,
            "properties": self.properties,
            "relationships": self.relationships,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "synonyms": self.synonyms,
            "namespace": self.namespace,
            "source_ontology": self.source_ontology,
            "confidence_score": self.confidence_score,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TMFEntity":
        entity = cls()
        entity.id = data.get("id", entity.id)
        entity.name = data.get("name", "")
        entity.description = data.get("description", "")
        entity.entity_type = EntityType(
            data.get("entity_type", EntityType.CONCEPT.value)
        )
        entity.properties = data.get("properties", {})
        entity.relationships = data.get("relationships", [])
        entity.parent_id = data.get("parent_id")
        entity.children_ids = data.get("children_ids", [])
        entity.synonyms = data.get("synonyms", [])
        entity.namespace = data.get("namespace", "")
        entity.source_ontology = data.get("source_ontology", "")
        entity.confidence_score = data.get("confidence_score", 1.0)
        return entity


@dataclass
class TMFOntology:
    name: str
    version: str = "1.0"
    namespace: str = ""
    entities: Dict[str, TMFEntity] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_entity(self, entity: TMFEntity) -> None:
        self.entities[entity.id] = entity

    def get_entity(self, entity_id: str) -> Optional[TMFEntity]:
        return self.entities.get(entity_id)

    def get_entities_by_type(self, entity_type: EntityType) -> List[TMFEntity]:
        return [
            entity
            for entity in self.entities.values()
            if entity.entity_type == entity_type
        ]

    def get_entities_by_name(self, name: str) -> List[TMFEntity]:
        return [
            entity
            for entity in self.entities.values()
            if entity.name.lower() == name.lower()
        ]

    def get_root_entities(self) -> List[TMFEntity]:
        return [entity for entity in self.entities.values() if entity.parent_id is None]

    def get_children(self, entity_id: str) -> List[TMFEntity]:
        entity = self.get_entity(entity_id)
        if not entity:
            return []
        return [
            self.entities[child_id]
            for child_id in entity.children_ids
            if child_id in self.entities
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "namespace": self.namespace,
            "entities": {
                eid: entity.to_dict() for eid, entity in self.entities.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TMFOntology":
        ontology = cls(
            name=data.get("name", ""),
            version=data.get("version", "1.0"),
            namespace=data.get("namespace", ""),
        )
        ontology.metadata = data.get("metadata", {})

        entities_data = data.get("entities", {})
        for entity_id, entity_data in entities_data.items():
            entity = TMFEntity.from_dict(entity_data)
            ontology.add_entity(entity)

        return ontology
