from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
import logging
from enum import Enum
import time

from ..entities.tmf_entity import TMFEntity, TMFOntology
from ..entities.knowledge_graph import TMFKnowledgeGraph
from ..matching.matching_engine import MatchingResult, TMFMatchingEngine


class AlignmentType(Enum):
    EQUIVALENCE = "equivalence"
    SUBSUMPTION = "subsumption"
    OVERLAP = "overlap"
    DISJOINT = "disjoint"


@dataclass
class AlignmentResult:
    source_entity: TMFEntity
    target_entity: TMFEntity
    alignment_type: AlignmentType
    confidence: float
    hierarchical_distance: float
    structural_similarity: float
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class AlignmentConfiguration:
    equivalence_threshold: float = 0.9
    subsumption_threshold: float = 0.7
    overlap_threshold: float = 0.5
    enable_hierarchical_analysis: bool = True
    max_hierarchical_depth: int = 5
    consider_sibling_relationships: bool = True
    enable_structural_propagation: bool = True
    confidence_decay_factor: float = 0.1


class HierarchicalAnalyzer:
    def __init__(self, config: AlignmentConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def analyze_hierarchical_relationship(self, entity1: TMFEntity, entity2: TMFEntity,
                                        kg1: TMFKnowledgeGraph, kg2: TMFKnowledgeGraph) -> Dict[str, Any]:
        
        ancestors1 = kg1.get_ancestors(entity1.id)
        ancestors2 = kg2.get_ancestors(entity2.id)
        descendants1 = kg1.get_descendants(entity1.id)
        descendants2 = kg2.get_descendants(entity2.id)
        
        hierarchical_distance = self._calculate_hierarchical_distance(
            entity1, entity2, kg1, kg2)
        
        common_ancestors = self._find_common_ancestor_patterns(ancestors1, ancestors2)
        common_descendants = self._find_common_descendant_patterns(descendants1, descendants2)
        
        level_difference = abs(len(ancestors1) - len(ancestors2))
        
        return {
            "hierarchical_distance": hierarchical_distance,
            "level_difference": level_difference,
            "common_ancestor_patterns": common_ancestors,
            "common_descendant_patterns": common_descendants,
            "ancestors_count": (len(ancestors1), len(ancestors2)),
            "descendants_count": (len(descendants1), len(descendants2))
        }
    
    def _calculate_hierarchical_distance(self, entity1: TMFEntity, entity2: TMFEntity,
                                       kg1: TMFKnowledgeGraph, kg2: TMFKnowledgeGraph) -> float:
        
        ancestors1 = set(e.name.lower() for e in kg1.get_ancestors(entity1.id))
        ancestors2 = set(e.name.lower() for e in kg2.get_ancestors(entity2.id))
        
        if not ancestors1 and not ancestors2:
            return 0.0
        
        if not ancestors1 or not ancestors2:
            return 1.0
        
        intersection = len(ancestors1 & ancestors2)
        union = len(ancestors1 | ancestors2)
        
        return 1.0 - (intersection / union) if union > 0 else 1.0
    
    def _find_common_ancestor_patterns(self, ancestors1: List[TMFEntity], 
                                     ancestors2: List[TMFEntity]) -> List[Tuple[str, str]]:
        patterns = []
        names1 = [a.name.lower() for a in ancestors1]
        names2 = [a.name.lower() for a in ancestors2]
        
        for name1 in names1:
            for name2 in names2:
                if self._are_similar_names(name1, name2):
                    patterns.append((name1, name2))
        
        return patterns
    
    def _find_common_descendant_patterns(self, descendants1: List[TMFEntity], 
                                       descendants2: List[TMFEntity]) -> List[Tuple[str, str]]:
        patterns = []
        names1 = [d.name.lower() for d in descendants1]
        names2 = [d.name.lower() for d in descendants2]
        
        for name1 in names1:
            for name2 in names2:
                if self._are_similar_names(name1, name2):
                    patterns.append((name1, name2))
        
        return patterns
    
    def _are_similar_names(self, name1: str, name2: str) -> bool:
        from difflib import SequenceMatcher
        return SequenceMatcher(None, name1, name2).ratio() > 0.7


class StructuralPropagator:
    def __init__(self, config: AlignmentConfiguration):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def propagate_alignments(self, initial_alignments: List[AlignmentResult],
                           kg1: TMFKnowledgeGraph, kg2: TMFKnowledgeGraph) -> List[AlignmentResult]:
        
        if not self.config.enable_structural_propagation:
            return initial_alignments
        
        propagated_alignments = initial_alignments.copy()
        alignment_map = {(a.source_entity.id, a.target_entity.id): a for a in initial_alignments}
        
        for alignment in initial_alignments:
            if alignment.confidence > 0.8:
                new_alignments = self._propagate_from_alignment(
                    alignment, kg1, kg2, alignment_map)
                propagated_alignments.extend(new_alignments)
        
        return self._deduplicate_alignments(propagated_alignments)
    
    def _propagate_from_alignment(self, base_alignment: AlignmentResult,
                                kg1: TMFKnowledgeGraph, kg2: TMFKnowledgeGraph,
                                existing_alignments: Dict[Tuple[str, str], AlignmentResult]) -> List[AlignmentResult]:
        
        new_alignments = []
        
        source_neighbors = kg1.get_neighbors(base_alignment.source_entity.id)
        target_neighbors = kg2.get_neighbors(base_alignment.target_entity.id)
        
        for source_neighbor in source_neighbors:
            for target_neighbor in target_neighbors:
                alignment_key = (source_neighbor.id, target_neighbor.id)
                
                if alignment_key not in existing_alignments:
                    propagated_confidence = base_alignment.confidence * (1 - self.config.confidence_decay_factor)
                    
                    if propagated_confidence > 0.3:
                        new_alignment = AlignmentResult(
                            source_entity=source_neighbor,
                            target_entity=target_neighbor,
                            alignment_type=AlignmentType.OVERLAP,
                            confidence=propagated_confidence,
                            hierarchical_distance=0.5,
                            structural_similarity=0.5,
                            details={
                                "propagated_from": base_alignment.source_entity.id,
                                "propagation_type": "neighbor"
                            }
                        )
                        new_alignments.append(new_alignment)
        
        return new_alignments
    
    def _deduplicate_alignments(self, alignments: List[AlignmentResult]) -> List[AlignmentResult]:
        unique_alignments = {}
        
        for alignment in alignments:
            key = (alignment.source_entity.id, alignment.target_entity.id)
            
            if key not in unique_alignments or alignment.confidence > unique_alignments[key].confidence:
                unique_alignments[key] = alignment
        
        return list(unique_alignments.values())


class TMFAlignmentEngine:
    def __init__(self, matching_engine: TMFMatchingEngine, 
                 config: AlignmentConfiguration = None):
        self.matching_engine = matching_engine
        self.config = config or AlignmentConfiguration()
        self.hierarchical_analyzer = HierarchicalAnalyzer(self.config)
        self.structural_propagator = StructuralPropagator(self.config)
        self.logger = logging.getLogger(__name__)
    
    def align_ontologies(self, source_ontology: TMFOntology, 
                        target_ontology: TMFOntology) -> List[AlignmentResult]:
        
        self.logger.info(f"Starting ontology alignment: {source_ontology.name} -> {target_ontology.name}")
        
        matching_results = self.matching_engine.match_ontologies(source_ontology, target_ontology)
        
        source_kg = TMFKnowledgeGraph(source_ontology)
        target_kg = TMFKnowledgeGraph(target_ontology)
        
        alignments = []
        
        for match in matching_results:
            alignment = self._create_alignment_from_match(match, source_kg, target_kg)
            if alignment:
                alignments.append(alignment)
        
        if self.config.enable_structural_propagation:
            alignments = self.structural_propagator.propagate_alignments(
                alignments, source_kg, target_kg)
        
        alignments = self._filter_and_rank_alignments(alignments)
        
        self.logger.info(f"Alignment completed. Generated {len(alignments)} alignments.")
        return alignments
    
    def _create_alignment_from_match(self, match: MatchingResult,
                                   source_kg: TMFKnowledgeGraph,
                                   target_kg: TMFKnowledgeGraph) -> Optional[AlignmentResult]:
        
        hierarchical_analysis = None
        if self.config.enable_hierarchical_analysis:
            hierarchical_analysis = self.hierarchical_analyzer.analyze_hierarchical_relationship(
                match.source_entity, match.target_entity, source_kg, target_kg)
        
        alignment_type = self._determine_alignment_type(
            match.similarity_score, hierarchical_analysis)
        
        hierarchical_distance = hierarchical_analysis.get("hierarchical_distance", 0.5) if hierarchical_analysis else 0.5
        
        structural_similarity = self._calculate_structural_similarity(
            match.source_entity, match.target_entity, source_kg, target_kg)
        
        adjusted_confidence = self._adjust_confidence_with_hierarchy(
            match.confidence, hierarchical_analysis)
        
        return AlignmentResult(
            source_entity=match.source_entity,
            target_entity=match.target_entity,
            alignment_type=alignment_type,
            confidence=adjusted_confidence,
            hierarchical_distance=hierarchical_distance,
            structural_similarity=structural_similarity,
            details={
                "original_match": {
                    "similarity_score": match.similarity_score,
                    "match_type": match.match_type,
                    "original_confidence": match.confidence
                },
                "hierarchical_analysis": hierarchical_analysis,
                "structural_features": self._extract_structural_features(
                    match.source_entity, match.target_entity, source_kg, target_kg)
            }
        )
    
    def _determine_alignment_type(self, similarity_score: float, 
                                hierarchical_analysis: Optional[Dict[str, Any]]) -> AlignmentType:
        
        if similarity_score >= self.config.equivalence_threshold:
            return AlignmentType.EQUIVALENCE
        elif similarity_score >= self.config.subsumption_threshold:
            if hierarchical_analysis:
                level_diff = hierarchical_analysis.get("level_difference", 0)
                if level_diff > 0:
                    return AlignmentType.SUBSUMPTION
            return AlignmentType.OVERLAP
        elif similarity_score >= self.config.overlap_threshold:
            return AlignmentType.OVERLAP
        else:
            return AlignmentType.DISJOINT
    
    def _calculate_structural_similarity(self, entity1: TMFEntity, entity2: TMFEntity,
                                       kg1: TMFKnowledgeGraph, kg2: TMFKnowledgeGraph) -> float:
        
        neighbors1 = set(n.name.lower() for n in kg1.get_neighbors(entity1.id))
        neighbors2 = set(n.name.lower() for n in kg2.get_neighbors(entity2.id))
        
        if not neighbors1 and not neighbors2:
            return 1.0
        if not neighbors1 or not neighbors2:
            return 0.0
        
        intersection = len(neighbors1 & neighbors2)
        union = len(neighbors1 | neighbors2)
        
        return intersection / union if union > 0 else 0.0
    
    def _adjust_confidence_with_hierarchy(self, original_confidence: float,
                                        hierarchical_analysis: Optional[Dict[str, Any]]) -> float:
        
        if not hierarchical_analysis:
            return original_confidence
        
        hierarchical_distance = hierarchical_analysis.get("hierarchical_distance", 0.5)
        level_difference = hierarchical_analysis.get("level_difference", 0)
        
        hierarchy_factor = 1.0 - (hierarchical_distance * 0.2)
        level_penalty = min(0.1, level_difference * 0.02)
        
        adjusted_confidence = original_confidence * hierarchy_factor - level_penalty
        
        return max(0.0, min(1.0, adjusted_confidence))
    
    def _extract_structural_features(self, entity1: TMFEntity, entity2: TMFEntity,
                                   kg1: TMFKnowledgeGraph, kg2: TMFKnowledgeGraph) -> Dict[str, Any]:
        
        return {
            "neighbor_count": (
                len(kg1.get_neighbors(entity1.id)),
                len(kg2.get_neighbors(entity2.id))
            ),
            "ancestor_count": (
                len(kg1.get_ancestors(entity1.id)),
                len(kg2.get_ancestors(entity2.id))
            ),
            "descendant_count": (
                len(kg1.get_descendants(entity1.id)),
                len(kg2.get_descendants(entity2.id))
            ),
            "property_count": (
                len(entity1.properties),
                len(entity2.properties)
            ),
            "relationship_count": (
                len(entity1.relationships),
                len(entity2.relationships)
            )
        }
    
    def _filter_and_rank_alignments(self, alignments: List[AlignmentResult]) -> List[AlignmentResult]:
        filtered_alignments = [a for a in alignments if a.confidence > 0.3]
        
        return sorted(filtered_alignments, 
                     key=lambda a: (a.confidence, a.structural_similarity), 
                     reverse=True)
    
    def get_alignment_statistics(self, alignments: List[AlignmentResult]) -> Dict[str, Any]:
        if not alignments:
            return {
                "total_alignments": 0,
                "alignment_types": {},
                "average_confidence": 0.0,
                "average_hierarchical_distance": 0.0,
                "average_structural_similarity": 0.0
            }
        
        total_alignments = len(alignments)
        
        alignment_types = {}
        for alignment in alignments:
            alignment_type = alignment.alignment_type.value
            alignment_types[alignment_type] = alignment_types.get(alignment_type, 0) + 1
        
        avg_confidence = sum(a.confidence for a in alignments) / total_alignments
        avg_hierarchical_distance = sum(a.hierarchical_distance for a in alignments) / total_alignments
        avg_structural_similarity = sum(a.structural_similarity for a in alignments) / total_alignments
        
        return {
            "total_alignments": total_alignments,
            "alignment_types": alignment_types,
            "average_confidence": avg_confidence,
            "average_hierarchical_distance": avg_hierarchical_distance,
            "average_structural_similarity": avg_structural_similarity,
            "confidence_distribution": self._calculate_distribution(
                [a.confidence for a in alignments]),
            "top_alignments": [
                {
                    "source": alignment.source_entity.name,
                    "target": alignment.target_entity.name,
                    "type": alignment.alignment_type.value,
                    "confidence": alignment.confidence,
                    "hierarchical_distance": alignment.hierarchical_distance,
                    "structural_similarity": alignment.structural_similarity
                }
                for alignment in alignments[:10]
            ]
        }
    
    def _calculate_distribution(self, values: List[float]) -> Dict[str, int]:
        ranges = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
        distribution = {f"{r[0]}-{r[1]}": 0 for r in ranges}
        
        for value in values:
            for range_start, range_end in ranges:
                if range_start <= value < range_end or (range_end == 1.0 and value == 1.0):
                    range_key = f"{range_start}-{range_end}"
                    distribution[range_key] += 1
                    break
        
        return distribution


class AlignmentResultProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def export_alignments_to_dict(self, alignments: List[AlignmentResult]) -> Dict[str, Any]:
        return {
            "alignments": [
                {
                    "source_entity": {
                        "id": alignment.source_entity.id,
                        "name": alignment.source_entity.name,
                        "type": alignment.source_entity.entity_type.value
                    },
                    "target_entity": {
                        "id": alignment.target_entity.id,
                        "name": alignment.target_entity.name,
                        "type": alignment.target_entity.entity_type.value
                    },
                    "alignment_type": alignment.alignment_type.value,
                    "confidence": alignment.confidence,
                    "hierarchical_distance": alignment.hierarchical_distance,
                    "structural_similarity": alignment.structural_similarity,
                    "timestamp": alignment.timestamp,
                    "details": alignment.details
                }
                for alignment in alignments
            ],
            "metadata": {
                "total_alignments": len(alignments),
                "generation_time": time.time()
            }
        }
    
    def create_alignment_mapping(self, alignments: List[AlignmentResult]) -> Dict[str, str]:
        mapping = {}
        
        for alignment in alignments:
            if alignment.alignment_type in [AlignmentType.EQUIVALENCE, AlignmentType.SUBSUMPTION]:
                source_id = alignment.source_entity.id
                target_id = alignment.target_entity.id
                
                if source_id not in mapping or alignment.confidence > 0.8:
                    mapping[source_id] = target_id
        
        return mapping
