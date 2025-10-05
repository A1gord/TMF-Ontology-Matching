from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
import networkx as nx
from .tmf_entity import TMFEntity, TMFOntology, EntityType


@dataclass
class KnowledgeGraphEdge:
    source_id: str
    target_id: str
    relationship_type: str
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


class TMFKnowledgeGraph:
    def __init__(self, ontology: TMFOntology):
        self.ontology = ontology
        self.graph = nx.DiGraph()
        self.edges: Dict[Tuple[str, str], KnowledgeGraphEdge] = {}
        self._build_graph()
    
    def _build_graph(self) -> None:
        for entity in self.ontology.entities.values():
            self.graph.add_node(entity.id, entity=entity)
            
            if entity.parent_id:
                self.add_edge(entity.parent_id, entity.id, "parent_of")
            
            for child_id in entity.children_ids:
                self.add_edge(entity.id, child_id, "parent_of")
            
            for related_id in entity.relationships:
                self.add_edge(entity.id, related_id, "related_to")
    
    def add_edge(self, source_id: str, target_id: str, 
                 relationship_type: str, weight: float = 1.0,
                 properties: Optional[Dict[str, Any]] = None) -> None:
        if properties is None:
            properties = {}
        
        edge = KnowledgeGraphEdge(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            weight=weight,
            properties=properties
        )
        
        self.edges[(source_id, target_id)] = edge
        self.graph.add_edge(source_id, target_id, 
                           relationship_type=relationship_type,
                           weight=weight, **properties)
    
    def get_neighbors(self, entity_id: str, 
                     relationship_type: Optional[str] = None) -> List[TMFEntity]:
        neighbors = []
        for neighbor_id in self.graph.neighbors(entity_id):
            edge_data = self.graph[entity_id][neighbor_id]
            if relationship_type is None or edge_data.get('relationship_type') == relationship_type:
                entity = self.ontology.get_entity(neighbor_id)
                if entity:
                    neighbors.append(entity)
        return neighbors
    
    def get_path(self, source_id: str, target_id: str) -> Optional[List[str]]:
        try:
            return nx.shortest_path(self.graph, source_id, target_id)
        except nx.NetworkXNoPath:
            return None
    
    def get_subgraph(self, entity_ids: Set[str]) -> 'TMFKnowledgeGraph':
        subgraph_ontology = TMFOntology(
            name=f"{self.ontology.name}_subgraph",
            version=self.ontology.version,
            namespace=self.ontology.namespace
        )
        
        for entity_id in entity_ids:
            entity = self.ontology.get_entity(entity_id)
            if entity:
                subgraph_ontology.add_entity(entity)
        
        return TMFKnowledgeGraph(subgraph_ontology)
    
    def get_ancestors(self, entity_id: str) -> List[TMFEntity]:
        ancestors = []
        current_entity = self.ontology.get_entity(entity_id)
        
        while current_entity and current_entity.parent_id:
            parent = self.ontology.get_entity(current_entity.parent_id)
            if parent:
                ancestors.append(parent)
                current_entity = parent
            else:
                break
        
        return ancestors
    
    def get_descendants(self, entity_id: str) -> List[TMFEntity]:
        descendants = []
        
        def _collect_descendants(eid: str):
            entity = self.ontology.get_entity(eid)
            if entity:
                for child_id in entity.children_ids:
                    child = self.ontology.get_entity(child_id)
                    if child:
                        descendants.append(child)
                        _collect_descendants(child_id)
        
        _collect_descendants(entity_id)
        return descendants
    
    def calculate_semantic_distance(self, entity1_id: str, entity2_id: str) -> float:
        path = self.get_path(entity1_id, entity2_id)
        if path is None:
            return float('inf')
        
        distance = 0.0
        for i in range(len(path) - 1):
            edge_key = (path[i], path[i + 1])
            if edge_key in self.edges:
                distance += 1.0 / self.edges[edge_key].weight
            else:
                distance += 1.0
        
        return distance
    
    def get_common_ancestors(self, entity1_id: str, entity2_id: str) -> List[TMFEntity]:
        ancestors1 = set(e.id for e in self.get_ancestors(entity1_id))
        ancestors2 = set(e.id for e in self.get_ancestors(entity2_id))
        
        common_ancestor_ids = ancestors1.intersection(ancestors2)
        return [self.ontology.get_entity(aid) for aid in common_ancestor_ids 
                if self.ontology.get_entity(aid)]
    
    def get_entity_context(self, entity_id: str, depth: int = 2) -> Dict[str, Any]:
        entity = self.ontology.get_entity(entity_id)
        if not entity:
            return {}
        
        context = {
            "entity": entity,
            "neighbors": self.get_neighbors(entity_id),
            "ancestors": self.get_ancestors(entity_id)[:depth],
            "descendants": self.get_descendants(entity_id)[:depth*2],
            "related_entities": []
        }
        
        for related_id in entity.relationships:
            related_entity = self.ontology.get_entity(related_id)
            if related_entity:
                context["related_entities"].append(related_entity)
        
        return context
    
    def export_to_networkx(self) -> nx.DiGraph:
        return self.graph.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        return {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "num_entities_by_type": {
                entity_type.value: len(self.ontology.get_entities_by_type(entity_type))
                for entity_type in EntityType
            },
            "average_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            "is_connected": nx.is_weakly_connected(self.graph),
            "num_connected_components": nx.number_weakly_connected_components(self.graph)
        }
