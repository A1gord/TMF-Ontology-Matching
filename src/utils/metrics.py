from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict

from ..matching.matching_engine import MatchingResult
from ..alignment.alignment_engine import AlignmentResult


@dataclass
class EvaluationMetrics:
    precision: float
    recall: float
    f1_score: float
    accuracy: Optional[float] = None
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0


@dataclass
class GroundTruthMapping:
    source_id: str
    target_id: str
    is_correct: bool
    mapping_type: str = "equivalence"


class MatchingEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_matching_results(
        self,
        matching_results: List[MatchingResult],
        ground_truth: List[GroundTruthMapping],
    ) -> EvaluationMetrics:

        predicted_mappings = self._extract_predicted_mappings(matching_results)
        gt_mappings = self._extract_ground_truth_mappings(ground_truth)

        tp = len(predicted_mappings & gt_mappings)
        fp = len(predicted_mappings - gt_mappings)
        fn = len(gt_mappings - predicted_mappings)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )

    def _extract_predicted_mappings(
        self, matching_results: List[MatchingResult]
    ) -> Set[Tuple[str, str]]:
        return {
            (result.source_entity.id, result.target_entity.id)
            for result in matching_results
        }

    def _extract_ground_truth_mappings(
        self, ground_truth: List[GroundTruthMapping]
    ) -> Set[Tuple[str, str]]:
        return {(gt.source_id, gt.target_id) for gt in ground_truth if gt.is_correct}

    def evaluate_by_threshold(
        self,
        matching_results: List[MatchingResult],
        ground_truth: List[GroundTruthMapping],
        thresholds: List[float],
    ) -> Dict[float, EvaluationMetrics]:

        results = {}

        for threshold in thresholds:
            filtered_results = [
                r for r in matching_results if r.similarity_score >= threshold
            ]
            metrics = self.evaluate_matching_results(filtered_results, ground_truth)
            results[threshold] = metrics

        return results

    def evaluate_by_match_type(
        self,
        matching_results: List[MatchingResult],
        ground_truth: List[GroundTruthMapping],
    ) -> Dict[str, EvaluationMetrics]:

        results = {}
        match_types = set(result.match_type for result in matching_results)

        for match_type in match_types:
            filtered_results = [
                r for r in matching_results if r.match_type == match_type
            ]
            metrics = self.evaluate_matching_results(filtered_results, ground_truth)
            results[match_type] = metrics

        return results

    def calculate_ranking_metrics(
        self,
        matching_results: List[MatchingResult],
        ground_truth: List[GroundTruthMapping],
        k_values: List[int] = None,
    ) -> Dict[str, float]:

        if k_values is None:
            k_values = [1, 5, 10]

        gt_mappings = self._extract_ground_truth_mappings(ground_truth)

        results_by_source = defaultdict(list)
        for result in matching_results:
            results_by_source[result.source_entity.id].append(result)

        for source_id in results_by_source:
            results_by_source[source_id].sort(
                key=lambda r: r.similarity_score, reverse=True
            )

        ranking_metrics = {}

        for k in k_values:
            hits_at_k = 0
            total_queries = len(results_by_source)

            for source_id, results in results_by_source.items():
                top_k_results = results[:k]

                for result in top_k_results:
                    if (
                        result.source_entity.id,
                        result.target_entity.id,
                    ) in gt_mappings:
                        hits_at_k += 1
                        break

            hits_at_k_rate = hits_at_k / total_queries if total_queries > 0 else 0.0
            ranking_metrics[f"hits_at_{k}"] = hits_at_k_rate

        mrr = self._calculate_mean_reciprocal_rank(results_by_source, gt_mappings)
        ranking_metrics["mean_reciprocal_rank"] = mrr

        return ranking_metrics

    def _calculate_mean_reciprocal_rank(
        self,
        results_by_source: Dict[str, List[MatchingResult]],
        gt_mappings: Set[Tuple[str, str]],
    ) -> float:

        reciprocal_ranks = []

        for source_id, results in results_by_source.items():
            for rank, result in enumerate(results, 1):
                if (result.source_entity.id, result.target_entity.id) in gt_mappings:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        return (
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        )


class AlignmentEvaluator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_alignment_results(
        self,
        alignment_results: List[AlignmentResult],
        ground_truth: List[GroundTruthMapping],
    ) -> EvaluationMetrics:

        predicted_mappings = self._extract_predicted_alignments(alignment_results)
        gt_mappings = self._extract_ground_truth_mappings(ground_truth)

        tp = len(predicted_mappings & gt_mappings)
        fp = len(predicted_mappings - gt_mappings)
        fn = len(gt_mappings - predicted_mappings)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return EvaluationMetrics(
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
        )

    def _extract_predicted_alignments(
        self, alignment_results: List[AlignmentResult]
    ) -> Set[Tuple[str, str]]:
        return {
            (result.source_entity.id, result.target_entity.id)
            for result in alignment_results
        }

    def _extract_ground_truth_mappings(
        self, ground_truth: List[GroundTruthMapping]
    ) -> Set[Tuple[str, str]]:
        return {(gt.source_id, gt.target_id) for gt in ground_truth if gt.is_correct}

    def evaluate_by_alignment_type(
        self,
        alignment_results: List[AlignmentResult],
        ground_truth: List[GroundTruthMapping],
    ) -> Dict[str, EvaluationMetrics]:

        results = {}
        alignment_types = set(
            result.alignment_type.value for result in alignment_results
        )

        for alignment_type in alignment_types:
            filtered_results = [
                r for r in alignment_results if r.alignment_type.value == alignment_type
            ]
            metrics = self.evaluate_alignment_results(filtered_results, ground_truth)
            results[alignment_type] = metrics

        return results

    def evaluate_hierarchical_consistency(
        self, alignment_results: List[AlignmentResult]
    ) -> Dict[str, float]:

        alignment_map = {}
        for result in alignment_results:
            alignment_map[result.source_entity.id] = result.target_entity.id

        consistency_violations = 0
        total_hierarchical_pairs = 0

        for result in alignment_results:
            source_entity = result.source_entity

            if source_entity.parent_id and source_entity.parent_id in alignment_map:
                total_hierarchical_pairs += 1

                target_entity = result.target_entity
                expected_parent_target = alignment_map[source_entity.parent_id]

                if target_entity.parent_id != expected_parent_target:
                    consistency_violations += 1

        consistency_rate = (
            1.0 - (consistency_violations / total_hierarchical_pairs)
            if total_hierarchical_pairs > 0
            else 1.0
        )

        return {
            "hierarchical_consistency_rate": consistency_rate,
            "total_hierarchical_pairs": total_hierarchical_pairs,
            "consistency_violations": consistency_violations,
        }


class ComprehensiveEvaluator:
    def __init__(self):
        self.matching_evaluator = MatchingEvaluator()
        self.alignment_evaluator = AlignmentEvaluator()
        self.logger = logging.getLogger(__name__)

    def evaluate_complete_pipeline(
        self,
        matching_results: List[MatchingResult],
        alignment_results: List[AlignmentResult],
        ground_truth: List[GroundTruthMapping],
    ) -> Dict[str, Any]:

        matching_metrics = self.matching_evaluator.evaluate_matching_results(
            matching_results, ground_truth
        )

        alignment_metrics = self.alignment_evaluator.evaluate_alignment_results(
            alignment_results, ground_truth
        )

        ranking_metrics = self.matching_evaluator.calculate_ranking_metrics(
            matching_results, ground_truth
        )

        hierarchical_metrics = (
            self.alignment_evaluator.evaluate_hierarchical_consistency(
                alignment_results
            )
        )

        return {
            "matching_evaluation": {
                "precision": matching_metrics.precision,
                "recall": matching_metrics.recall,
                "f1_score": matching_metrics.f1_score,
                "true_positives": matching_metrics.true_positives,
                "false_positives": matching_metrics.false_positives,
                "false_negatives": matching_metrics.false_negatives,
            },
            "alignment_evaluation": {
                "precision": alignment_metrics.precision,
                "recall": alignment_metrics.recall,
                "f1_score": alignment_metrics.f1_score,
                "true_positives": alignment_metrics.true_positives,
                "false_positives": alignment_metrics.false_positives,
                "false_negatives": alignment_metrics.false_negatives,
            },
            "ranking_metrics": ranking_metrics,
            "hierarchical_consistency": hierarchical_metrics,
            "summary": {
                "overall_f1": (matching_metrics.f1_score + alignment_metrics.f1_score)
                / 2,
                "overall_precision": (
                    matching_metrics.precision + alignment_metrics.precision
                )
                / 2,
                "overall_recall": (matching_metrics.recall + alignment_metrics.recall)
                / 2,
            },
        }

    def generate_evaluation_report(self, evaluation_results: Dict[str, Any]) -> str:

        report_lines = [
            "=== TMF Ontology Matching and Alignment Evaluation Report ===",
            "",
            "MATCHING RESULTS:",
            f"  Precision: {evaluation_results['matching_evaluation']['precision']:.4f}",
            f"  Recall: {evaluation_results['matching_evaluation']['recall']:.4f}",
            f"  F1-Score: {evaluation_results['matching_evaluation']['f1_score']:.4f}",
            f"  True Positives: {evaluation_results['matching_evaluation']['true_positives']}",
            f"  False Positives: {evaluation_results['matching_evaluation']['false_positives']}",
            f"  False Negatives: {evaluation_results['matching_evaluation']['false_negatives']}",
            "",
            "ALIGNMENT RESULTS:",
            f"  Precision: {evaluation_results['alignment_evaluation']['precision']:.4f}",
            f"  Recall: {evaluation_results['alignment_evaluation']['recall']:.4f}",
            f"  F1-Score: {evaluation_results['alignment_evaluation']['f1_score']:.4f}",
            f"  True Positives: {evaluation_results['alignment_evaluation']['true_positives']}",
            f"  False Positives: {evaluation_results['alignment_evaluation']['false_positives']}",
            f"  False Negatives: {evaluation_results['alignment_evaluation']['false_negatives']}",
            "",
            "RANKING METRICS:",
        ]

        for metric, value in evaluation_results["ranking_metrics"].items():
            report_lines.append(f"  {metric}: {value:.4f}")

        report_lines.extend(
            [
                "",
                "HIERARCHICAL CONSISTENCY:",
                f"  Consistency Rate: {evaluation_results['hierarchical_consistency']['hierarchical_consistency_rate']:.4f}",
                f"  Total Hierarchical Pairs: {evaluation_results['hierarchical_consistency']['total_hierarchical_pairs']}",
                f"  Consistency Violations: {evaluation_results['hierarchical_consistency']['consistency_violations']}",
                "",
                "OVERALL SUMMARY:",
                f"  Overall F1-Score: {evaluation_results['summary']['overall_f1']:.4f}",
                f"  Overall Precision: {evaluation_results['summary']['overall_precision']:.4f}",
                f"  Overall Recall: {evaluation_results['summary']['overall_recall']:.4f}",
                "",
            ]
        )

        return "\n".join(report_lines)


class MetricsCalculator:
    @staticmethod
    def calculate_precision(true_positives: int, false_positives: int) -> float:
        return (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )

    @staticmethod
    def calculate_recall(true_positives: int, false_negatives: int) -> float:
        return (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0.0
        )

    @staticmethod
    def calculate_f1_score(precision: float, recall: float) -> float:
        return (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    @staticmethod
    def calculate_accuracy(
        true_positives: int,
        true_negatives: int,
        false_positives: int,
        false_negatives: int,
    ) -> float:
        total = true_positives + true_negatives + false_positives + false_negatives
        return (true_positives + true_negatives) / total if total > 0 else 0.0

    @staticmethod
    def calculate_jaccard_index(set1: Set[Any], set2: Set[Any]) -> float:
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0
