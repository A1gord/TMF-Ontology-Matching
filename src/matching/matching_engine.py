from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from ..entities.tmf_entity import TMFEntity, TMFOntology
from ..entities.knowledge_graph import TMFKnowledgeGraph
from .comparison_strategies import ComparisonStrategy, ComparisonResult, CompositeComparisonStrategy


@dataclass
class MatchingResult:
    source_entity: TMFEntity
    target_entity: TMFEntity
    similarity_score: float
    confidence: float
    match_type: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


@dataclass
class MatchingConfiguration:
    similarity_threshold: float = 0.6
    confidence_threshold: float = 0.5
    max_matches_per_entity: int = 5
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_context_matching: bool = True
    context_depth: int = 2
    filter_by_entity_type: bool = True
    enable_bidirectional_matching: bool = False


class TMFMatchingEngine:
    def __init__(self, comparison_strategy: ComparisonStrategy, 
                 config: MatchingConfiguration = None):
        self.comparison_strategy = comparison_strategy
        self.config = config or MatchingConfiguration()
        self.logger = logging.getLogger(__name__)
    
    def match_ontologies(self, source_ontology: TMFOntology, 
                        target_ontology: TMFOntology) -> List[MatchingResult]:
        self.logger.info(f"Starting ontology matching: {source_ontology.name} -> {target_ontology.name}")
        
        source_kg = TMFKnowledgeGraph(source_ontology)
        target_kg = TMFKnowledgeGraph(target_ontology)
        
        matches = []
        
        if self.config.enable_parallel_processing:
            matches = self._match_ontologies_parallel(source_kg, target_kg)
        else:
            matches = self._match_ontologies_sequential(source_kg, target_kg)
        
        matches = self._filter_matches(matches)
        matches = self._rank_matches(matches)
        
        self.logger.info(f"Matching completed. Found {len(matches)} matches.")
        return matches
    
    def _match_ontologies_parallel(self, source_kg: TMFKnowledgeGraph, 
                                  target_kg: TMFKnowledgeGraph) -> List[MatchingResult]:
        matches = []
        source_entities = list(source_kg.ontology.entities.values())
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_entity = {
                executor.submit(self._match_entity, entity, source_kg, target_kg): entity
                for entity in source_entities
            }
            
            for future in as_completed(future_to_entity):
                entity = future_to_entity[future]
                try:
                    entity_matches = future.result()
                    matches.extend(entity_matches)
                except Exception as e:
                    self.logger.error(f"Error matching entity {entity.id}: {e}")
        
        return matches
    
    def _match_ontologies_sequential(self, source_kg: TMFKnowledgeGraph, 
                                   target_kg: TMFKnowledgeGraph) -> List[MatchingResult]:
        matches = []
        
        for source_entity in source_kg.ontology.entities.values():
            entity_matches = self._match_entity(source_entity, source_kg, target_kg)
            matches.extend(entity_matches)
        
        return matches
    
    def _match_entity(self, source_entity: TMFEntity, 
                     source_kg: TMFKnowledgeGraph, 
                     target_kg: TMFKnowledgeGraph) -> List[MatchingResult]:
        matches = []
        
        target_entities = self._get_candidate_entities(source_entity, target_kg)
        
        source_context = None
        if self.config.enable_context_matching:
            source_context = source_kg.get_entity_context(
                source_entity.id, self.config.context_depth)
        
        for target_entity in target_entities:
            target_context = None
            if self.config.enable_context_matching:
                target_context = target_kg.get_entity_context(
                    target_entity.id, self.config.context_depth)
            
            comparison_result = self.comparison_strategy.compare(
                source_entity, target_entity, source_context, target_context)
            
            if (comparison_result.similarity_score >= self.config.similarity_threshold and
                comparison_result.confidence >= self.config.confidence_threshold):
                
                match = MatchingResult(
                    source_entity=source_entity,
                    target_entity=target_entity,
                    similarity_score=comparison_result.similarity_score,
                    confidence=comparison_result.confidence,
                    match_type=comparison_result.match_type,
                    details=comparison_result.details
                )
                matches.append(match)
        
        return matches
    
    def _get_candidate_entities(self, source_entity: TMFEntity, 
                              target_kg: TMFKnowledgeGraph) -> List[TMFEntity]:
        if self.config.filter_by_entity_type:
            candidates = target_kg.ontology.get_entities_by_type(source_entity.entity_type)
        else:
            candidates = list(target_kg.ontology.entities.values())
        
        return candidates
    
    def _filter_matches(self, matches: List[MatchingResult]) -> List[MatchingResult]:
        filtered_matches = []
        
        entity_matches = {}
        for match in matches:
            source_id = match.source_entity.id
            if source_id not in entity_matches:
                entity_matches[source_id] = []
            entity_matches[source_id].append(match)
        
        for source_id, entity_match_list in entity_matches.items():
            entity_match_list.sort(key=lambda m: m.similarity_score, reverse=True)
            top_matches = entity_match_list[:self.config.max_matches_per_entity]
            filtered_matches.extend(top_matches)
        
        return filtered_matches
    
    def _rank_matches(self, matches: List[MatchingResult]) -> List[MatchingResult]:
        return sorted(matches, key=lambda m: (m.similarity_score, m.confidence), reverse=True)
    
    def match_entities(self, source_entities: List[TMFEntity], 
                      target_entities: List[TMFEntity]) -> List[MatchingResult]:
        matches = []
        
        for source_entity in source_entities:
            for target_entity in target_entities:
                if (self.config.filter_by_entity_type and 
                    source_entity.entity_type != target_entity.entity_type):
                    continue
                
                comparison_result = self.comparison_strategy.compare(
                    source_entity, target_entity)
                
                if (comparison_result.similarity_score >= self.config.similarity_threshold and
                    comparison_result.confidence >= self.config.confidence_threshold):
                    
                    match = MatchingResult(
                        source_entity=source_entity,
                        target_entity=target_entity,
                        similarity_score=comparison_result.similarity_score,
                        confidence=comparison_result.confidence,
                        match_type=comparison_result.match_type,
                        details=comparison_result.details
                    )
                    matches.append(match)
        
        return self._rank_matches(matches)
    
    def get_matching_statistics(self, matches: List[MatchingResult]) -> Dict[str, Any]:
        if not matches:
            return {
                "total_matches": 0,
                "average_similarity": 0.0,
                "average_confidence": 0.0,
                "match_types": {},
                "similarity_distribution": {},
                "confidence_distribution": {}
            }
        
        total_matches = len(matches)
        avg_similarity = sum(m.similarity_score for m in matches) / total_matches
        avg_confidence = sum(m.confidence for m in matches) / total_matches
        
        match_types = {}
        for match in matches:
            match_type = match.match_type
            match_types[match_type] = match_types.get(match_type, 0) + 1
        
        similarity_ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.0)]
        similarity_distribution = {f"{r[0]}-{r[1]}": 0 for r in similarity_ranges}
        
        for match in matches:
            for range_start, range_end in similarity_ranges:
                if range_start <= match.similarity_score < range_end:
                    range_key = f"{range_start}-{range_end}"
                    similarity_distribution[range_key] += 1
                    break
        
        confidence_distribution = {f"{r[0]}-{r[1]}": 0 for r in similarity_ranges}
        for match in matches:
            for range_start, range_end in similarity_ranges:
                if range_start <= match.confidence < range_end:
                    range_key = f"{range_start}-{range_end}"
                    confidence_distribution[range_key] += 1
                    break
        
        return {
            "total_matches": total_matches,
            "average_similarity": avg_similarity,
            "average_confidence": avg_confidence,
            "match_types": match_types,
            "similarity_distribution": similarity_distribution,
            "confidence_distribution": confidence_distribution,
            "top_matches": [
                {
                    "source": match.source_entity.name,
                    "target": match.target_entity.name,
                    "similarity": match.similarity_score,
                    "confidence": match.confidence,
                    "type": match.match_type
                }
                for match in matches[:10]
            ]
        }


class MatchingResultProcessor:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def create_mapping_dict(self, matches: List[MatchingResult]) -> Dict[str, str]:
        mapping = {}
        for match in matches:
            source_id = match.source_entity.id
            target_id = match.target_entity.id
            
            if source_id not in mapping:
                mapping[source_id] = target_id
            else:
                existing_match_score = next(
                    (m.similarity_score for m in matches 
                     if m.source_entity.id == source_id and m.target_entity.id == mapping[source_id]),
                    0.0
                )
                if match.similarity_score > existing_match_score:
                    mapping[source_id] = target_id
        
        return mapping
    
    def export_matches_to_dict(self, matches: List[MatchingResult]) -> Dict[str, Any]:
        return {
            "matches": [
                {
                    "source_entity": {
                        "id": match.source_entity.id,
                        "name": match.source_entity.name,
                        "type": match.source_entity.entity_type.value
                    },
                    "target_entity": {
                        "id": match.target_entity.id,
                        "name": match.target_entity.name,
                        "type": match.target_entity.entity_type.value
                    },
                    "similarity_score": match.similarity_score,
                    "confidence": match.confidence,
                    "match_type": match.match_type,
                    "timestamp": match.timestamp,
                    "details": match.details
                }
                for match in matches
            ],
            "metadata": {
                "total_matches": len(matches),
                "generation_time": time.time()
            }
        }
    
    def filter_matches_by_threshold(self, matches: List[MatchingResult], 
                                  similarity_threshold: float = None,
                                  confidence_threshold: float = None) -> List[MatchingResult]:
        filtered = matches
        
        if similarity_threshold is not None:
            filtered = [m for m in filtered if m.similarity_score >= similarity_threshold]
        
        if confidence_threshold is not None:
            filtered = [m for m in filtered if m.confidence >= confidence_threshold]
        
        return filtered
    
    def group_matches_by_source(self, matches: List[MatchingResult]) -> Dict[str, List[MatchingResult]]:
        grouped = {}
        for match in matches:
            source_id = match.source_entity.id
            if source_id not in grouped:
                grouped[source_id] = []
            grouped[source_id].append(match)
        
        for source_id in grouped:
            grouped[source_id].sort(key=lambda m: m.similarity_score, reverse=True)
        
        return grouped
