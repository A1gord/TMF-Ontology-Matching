from abc import ABC, abstractmethod
from typing import List, Dict, Any, Set
import re
from difflib import SequenceMatcher
import logging
from dataclasses import dataclass

from ..entities.tmf_entity import TMFEntity, TMFOntology
from ..entities.knowledge_graph import TMFKnowledgeGraph


@dataclass
class ComparisonResult:
    similarity_score: float
    confidence: float
    match_type: str
    details: Dict[str, Any]


class ComparisonStrategy(ABC):
    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.logger = logging.getLogger(__name__)
    
    @abstractmethod
    def compare(self, entity1: TMFEntity, entity2: TMFEntity, 
                context1: Dict[str, Any] = None, 
                context2: Dict[str, Any] = None) -> ComparisonResult:
        pass


class ExactMatchStrategy(ComparisonStrategy):
    def __init__(self, weight: float = 1.0, case_sensitive: bool = False):
        super().__init__(weight)
        self.case_sensitive = case_sensitive
    
    def compare(self, entity1: TMFEntity, entity2: TMFEntity, 
                context1: Dict[str, Any] = None, 
                context2: Dict[str, Any] = None) -> ComparisonResult:
        
        name1 = entity1.name if self.case_sensitive else entity1.name.lower()
        name2 = entity2.name if self.case_sensitive else entity2.name.lower()
        
        if name1 == name2:
            return ComparisonResult(
                similarity_score=1.0,
                confidence=1.0,
                match_type="exact_match",
                details={"matched_names": [entity1.name, entity2.name]}
            )
        
        return ComparisonResult(
            similarity_score=0.0,
            confidence=1.0,
            match_type="no_match",
            details={}
        )


class SynonymMatchStrategy(ComparisonStrategy):
    def __init__(self, weight: float = 0.9, synonym_threshold: float = 0.8):
        super().__init__(weight)
        self.synonym_threshold = synonym_threshold
    
    def compare(self, entity1: TMFEntity, entity2: TMFEntity, 
                context1: Dict[str, Any] = None, 
                context2: Dict[str, Any] = None) -> ComparisonResult:
        
        all_names1 = {entity1.name.lower()} | {s.lower() for s in entity1.synonyms}
        all_names2 = {entity2.name.lower()} | {s.lower() for s in entity2.synonyms}
        
        best_score = 0.0
        best_match = None
        
        for name1 in all_names1:
            for name2 in all_names2:
                score = self._calculate_string_similarity(name1, name2)
                if score > best_score:
                    best_score = score
                    best_match = (name1, name2)
        
        if best_score >= self.synonym_threshold:
            return ComparisonResult(
                similarity_score=best_score,
                confidence=0.8,
                match_type="synonym_match",
                details={
                    "matched_terms": best_match,
                    "all_synonyms1": list(all_names1),
                    "all_synonyms2": list(all_names2)
                }
            )
        
        return ComparisonResult(
            similarity_score=best_score,
            confidence=0.6,
            match_type="partial_match" if best_score > 0.3 else "no_match",
            details={"best_match": best_match, "score": best_score}
        )
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        return SequenceMatcher(None, str1, str2).ratio()


class StructuralSimilarityStrategy(ComparisonStrategy):
    def __init__(self, weight: float = 0.7):
        super().__init__(weight)
    
    def compare(self, entity1: TMFEntity, entity2: TMFEntity, 
                context1: Dict[str, Any] = None, 
                context2: Dict[str, Any] = None) -> ComparisonResult:
        
        scores = {}
        
        scores['type_similarity'] = 1.0 if entity1.entity_type == entity2.entity_type else 0.0
        
        scores['property_similarity'] = self._compare_properties(entity1.properties, entity2.properties)
        
        scores['relationship_similarity'] = self._compare_relationships(
            entity1.relationships, entity2.relationships)
        
        scores['hierarchy_similarity'] = self._compare_hierarchy_position(entity1, entity2)
        
        if context1 and context2:
            scores['context_similarity'] = self._compare_contexts(context1, context2)
        else:
            scores['context_similarity'] = 0.0
        
        overall_score = (
            scores['type_similarity'] * 0.3 +
            scores['property_similarity'] * 0.25 +
            scores['relationship_similarity'] * 0.2 +
            scores['hierarchy_similarity'] * 0.15 +
            scores['context_similarity'] * 0.1
        )
        
        confidence = min(0.9, overall_score + 0.1)
        
        match_type = "structural_match" if overall_score > 0.6 else \
                    "partial_structural_match" if overall_score > 0.3 else "no_match"
        
        return ComparisonResult(
            similarity_score=overall_score,
            confidence=confidence,
            match_type=match_type,
            details=scores
        )
    
    def _compare_properties(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> float:
        if not props1 and not props2:
            return 1.0
        if not props1 or not props2:
            return 0.0
        
        common_keys = set(props1.keys()) & set(props2.keys())
        all_keys = set(props1.keys()) | set(props2.keys())
        
        if not all_keys:
            return 1.0
        
        key_similarity = len(common_keys) / len(all_keys)
        
        value_similarities = []
        for key in common_keys:
            val1, val2 = str(props1[key]), str(props2[key])
            value_similarities.append(SequenceMatcher(None, val1, val2).ratio())
        
        value_similarity = sum(value_similarities) / len(value_similarities) if value_similarities else 0.0
        
        return (key_similarity + value_similarity) / 2
    
    def _compare_relationships(self, rels1: List[str], rels2: List[str]) -> float:
        if not rels1 and not rels2:
            return 1.0
        if not rels1 or not rels2:
            return 0.0
        
        set1, set2 = set(rels1), set(rels2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _compare_hierarchy_position(self, entity1: TMFEntity, entity2: TMFEntity) -> float:
        score = 0.0
        
        if entity1.parent_id and entity2.parent_id:
            score += 0.5
        elif not entity1.parent_id and not entity2.parent_id:
            score += 0.5
        
        children_similarity = self._compare_relationships(entity1.children_ids, entity2.children_ids)
        score += children_similarity * 0.5
        
        return score
    
    def _compare_contexts(self, context1: Dict[str, Any], context2: Dict[str, Any]) -> float:
        scores = []
        
        neighbors1 = context1.get('neighbors', [])
        neighbors2 = context2.get('neighbors', [])
        neighbor_names1 = {n.name.lower() for n in neighbors1}
        neighbor_names2 = {n.name.lower() for n in neighbors2}
        
        if neighbor_names1 or neighbor_names2:
            neighbor_similarity = len(neighbor_names1 & neighbor_names2) / \
                                len(neighbor_names1 | neighbor_names2) if \
                                (neighbor_names1 | neighbor_names2) else 0.0
            scores.append(neighbor_similarity)
        
        ancestors1 = context1.get('ancestors', [])
        ancestors2 = context2.get('ancestors', [])
        ancestor_names1 = {a.name.lower() for a in ancestors1}
        ancestor_names2 = {a.name.lower() for a in ancestors2}
        
        if ancestor_names1 or ancestor_names2:
            ancestor_similarity = len(ancestor_names1 & ancestor_names2) / \
                                len(ancestor_names1 | ancestor_names2) if \
                                (ancestor_names1 | ancestor_names2) else 0.0
            scores.append(ancestor_similarity)
        
        return sum(scores) / len(scores) if scores else 0.0


class SemanticSimilarityStrategy(ComparisonStrategy):
    def __init__(self, weight: float = 0.8):
        super().__init__(weight)
        self.word_patterns = {
            'service': ['service', 'svc', 'srv'],
            'management': ['management', 'mgmt', 'manage'],
            'configuration': ['configuration', 'config', 'cfg'],
            'interface': ['interface', 'api', 'endpoint'],
            'process': ['process', 'workflow', 'procedure'],
            'model': ['model', 'schema', 'structure']
        }
    
    def compare(self, entity1: TMFEntity, entity2: TMFEntity, 
                context1: Dict[str, Any] = None, 
                context2: Dict[str, Any] = None) -> ComparisonResult:
        
        text1 = self._extract_text_features(entity1)
        text2 = self._extract_text_features(entity2)
        
        word_similarity = self._calculate_word_similarity(text1, text2)
        pattern_similarity = self._calculate_pattern_similarity(text1, text2)
        description_similarity = self._calculate_description_similarity(
            entity1.description, entity2.description)
        
        overall_score = (
            word_similarity * 0.4 +
            pattern_similarity * 0.3 +
            description_similarity * 0.3
        )
        
        confidence = min(0.85, overall_score + 0.15)
        
        match_type = "semantic_match" if overall_score > 0.7 else \
                    "partial_semantic_match" if overall_score > 0.4 else "no_match"
        
        return ComparisonResult(
            similarity_score=overall_score,
            confidence=confidence,
            match_type=match_type,
            details={
                "word_similarity": word_similarity,
                "pattern_similarity": pattern_similarity,
                "description_similarity": description_similarity,
                "text_features1": text1,
                "text_features2": text2
            }
        )
    
    def _extract_text_features(self, entity: TMFEntity) -> Set[str]:
        text = f"{entity.name} {entity.description} {' '.join(entity.synonyms)}"
        words = re.findall(r'\b\w+\b', text.lower())
        return set(words)
    
    def _calculate_word_similarity(self, words1: Set[str], words2: Set[str]) -> float:
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_pattern_similarity(self, words1: Set[str], words2: Set[str]) -> float:
        patterns1 = self._identify_patterns(words1)
        patterns2 = self._identify_patterns(words2)
        
        if not patterns1 and not patterns2:
            return 0.5
        if not patterns1 or not patterns2:
            return 0.0
        
        common_patterns = len(patterns1 & patterns2)
        total_patterns = len(patterns1 | patterns2)
        
        return common_patterns / total_patterns if total_patterns > 0 else 0.0
    
    def _identify_patterns(self, words: Set[str]) -> Set[str]:
        patterns = set()
        for word in words:
            for pattern, variants in self.word_patterns.items():
                if word in variants:
                    patterns.add(pattern)
        return patterns
    
    def _calculate_description_similarity(self, desc1: str, desc2: str) -> float:
        if not desc1 and not desc2:
            return 1.0
        if not desc1 or not desc2:
            return 0.0
        
        return SequenceMatcher(None, desc1.lower(), desc2.lower()).ratio()


class CompositeComparisonStrategy(ComparisonStrategy):
    def __init__(self, strategies: List[ComparisonStrategy]):
        super().__init__(1.0)
        self.strategies = strategies
        total_weight = sum(s.weight for s in strategies)
        if total_weight > 0:
            for strategy in self.strategies:
                strategy.weight = strategy.weight / total_weight
    
    def compare(self, entity1: TMFEntity, entity2: TMFEntity, 
                context1: Dict[str, Any] = None, 
                context2: Dict[str, Any] = None) -> ComparisonResult:
        
        results = []
        weighted_score = 0.0
        weighted_confidence = 0.0
        
        for strategy in self.strategies:
            result = strategy.compare(entity1, entity2, context1, context2)
            results.append(result)
            weighted_score += result.similarity_score * strategy.weight
            weighted_confidence += result.confidence * strategy.weight
        
        best_result = max(results, key=lambda r: r.similarity_score)
        
        return ComparisonResult(
            similarity_score=weighted_score,
            confidence=weighted_confidence,
            match_type=best_result.match_type,
            details={
                "individual_results": [
                    {
                        "strategy": type(strategy).__name__,
                        "weight": strategy.weight,
                        "result": result.__dict__
                    }
                    for strategy, result in zip(self.strategies, results)
                ],
                "weighted_score": weighted_score,
                "best_individual_score": best_result.similarity_score
            }
        )
