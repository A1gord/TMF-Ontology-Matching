import argparse
import sys
from pathlib import Path
import logging

from utils.config import ConfigManager, TMFConfig, setup_logging, EnvironmentConfigLoader, ConfigValidator
from utils.xml_parser import UniversalTMFParser
from utils.serialization import UniversalSerializer
from utils.metrics import ComprehensiveEvaluator, GroundTruthMapping
from utils.neo4j_handler import Neo4jConnector, TMFNeo4jExporter, TMFNeo4jImporter
from utils.rdf_converter import RDFConverter, Neo4jToRDFConverter
from matching.comparison_strategies import (
    ExactMatchStrategy, SynonymMatchStrategy, StructuralSimilarityStrategy,
    SemanticSimilarityStrategy, CompositeComparisonStrategy
)
from matching.matching_engine import TMFMatchingEngine, MatchingConfiguration
from alignment.alignment_engine import TMFAlignmentEngine, AlignmentConfiguration


class TMFOntologyMatcher:
    def __init__(self, config_path: str = None):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load_config()
        
        env_loader = EnvironmentConfigLoader()
        env_updates = env_loader.load_from_environment()
        if env_updates:
            self.config_manager.update_config(env_updates)
            self.config = self.config_manager.get_config()
        
        setup_logging(self.config.logging)
        self.logger = logging.getLogger(__name__)
        
        validator = ConfigValidator()
        validation_errors = validator.validate_config(self.config)
        if validation_errors:
            self.logger.error("Configuration validation errors:")
            for error in validation_errors:
                self.logger.error(f"  - {error}")
            sys.exit(1)
        
        self.parser = UniversalTMFParser(self.config)
        self.serializer = UniversalSerializer(self.config)
        self.evaluator = ComprehensiveEvaluator()
        self.rdf_converter = RDFConverter(self.config)
        
        self.neo4j_connector = None
        self.neo4j_exporter = None
        self.neo4j_importer = None
        self.neo4j_rdf_converter = None
        
        if self.config.neo4j.enabled:
            self._setup_neo4j()
        
        self._setup_matching_engine()
        self._setup_alignment_engine()
    
    def _setup_matching_engine(self):
        strategies = []
        
        for strategy_config in self.config.matching.comparison_strategies:
            strategy = self._create_strategy(strategy_config)
            if strategy:
                strategies.append(strategy)
        
        if not strategies:
            strategies = [
                ExactMatchStrategy(weight=1.0),
                SynonymMatchStrategy(weight=0.9),
                StructuralSimilarityStrategy(weight=0.7),
                SemanticSimilarityStrategy(weight=0.8)
            ]
        
        composite_strategy = CompositeComparisonStrategy(strategies)
        
        matching_config = MatchingConfiguration(
            similarity_threshold=self.config.matching.similarity_threshold,
            confidence_threshold=self.config.matching.confidence_threshold,
            max_matches_per_entity=self.config.matching.max_matches_per_entity,
            enable_parallel_processing=self.config.matching.enable_parallel_processing,
            max_workers=self.config.matching.max_workers,
            enable_context_matching=self.config.matching.enable_context_matching,
            context_depth=self.config.matching.context_depth,
            filter_by_entity_type=self.config.matching.filter_by_entity_type,
            enable_bidirectional_matching=self.config.matching.enable_bidirectional_matching
        )
        
        self.matching_engine = TMFMatchingEngine(composite_strategy, matching_config)
    
    def _setup_alignment_engine(self):
        alignment_config = AlignmentConfiguration(
            equivalence_threshold=self.config.alignment.equivalence_threshold,
            subsumption_threshold=self.config.alignment.subsumption_threshold,
            overlap_threshold=self.config.alignment.overlap_threshold,
            enable_hierarchical_analysis=self.config.alignment.enable_hierarchical_analysis,
            max_hierarchical_depth=self.config.alignment.max_hierarchical_depth,
            consider_sibling_relationships=self.config.alignment.consider_sibling_relationships,
            enable_structural_propagation=self.config.alignment.enable_structural_propagation,
            confidence_decay_factor=self.config.alignment.confidence_decay_factor
        )
        
        self.alignment_engine = TMFAlignmentEngine(self.matching_engine, alignment_config)
    
    def _setup_neo4j(self):
        try:
            self.neo4j_connector = Neo4jConnector(self.config.neo4j)
            self.neo4j_exporter = TMFNeo4jExporter(self.neo4j_connector)
            self.neo4j_importer = TMFNeo4jImporter(self.neo4j_connector)
            self.neo4j_rdf_converter = Neo4jToRDFConverter(self.neo4j_connector)
            self.logger.info("Neo4j components initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Neo4j: {e}")
            self.config.neo4j.enabled = False
    
    def _create_strategy(self, strategy_config):
        strategy_classes = {
            'ExactMatchStrategy': ExactMatchStrategy,
            'SynonymMatchStrategy': SynonymMatchStrategy,
            'StructuralSimilarityStrategy': StructuralSimilarityStrategy,
            'SemanticSimilarityStrategy': SemanticSimilarityStrategy
        }
        
        strategy_class = strategy_classes.get(strategy_config.strategy_type)
        if not strategy_class:
            self.logger.warning(f"Unknown strategy type: {strategy_config.strategy_type}")
            return None
        
        try:
            return strategy_class(weight=strategy_config.weight, **strategy_config.parameters)
        except Exception as e:
            self.logger.error(f"Error creating strategy {strategy_config.strategy_type}: {e}")
            return None
    
    def match_ontologies(self, source_path: str, target_path: str, output_path: str = None):
        try:
            self.logger.info(f"Loading source ontology from {source_path}")
            source_ontology = self.parser.parse_file(source_path)
            
            self.logger.info(f"Loading target ontology from {target_path}")
            target_ontology = self.parser.parse_file(target_path)
            
            self.logger.info("Starting matching process")
            matching_results = self.matching_engine.match_ontologies(source_ontology, target_ontology)
            
            matching_stats = self.matching_engine.get_matching_statistics(matching_results)
            self.logger.info(f"Matching completed: {matching_stats['total_matches']} matches found")
            
            if output_path:
                results_data = {
                    'matches': [
                        {
                            'source_entity': {
                                'id': match.source_entity.id,
                                'name': match.source_entity.name,
                                'type': match.source_entity.entity_type.value
                            },
                            'target_entity': {
                                'id': match.target_entity.id,
                                'name': match.target_entity.name,
                                'type': match.target_entity.entity_type.value
                            },
                            'similarity_score': match.similarity_score,
                            'confidence': match.confidence,
                            'match_type': match.match_type,
                            'details': match.details
                        }
                        for match in matching_results
                    ],
                    'statistics': matching_stats,
                    'metadata': {
                        'source_ontology': source_ontology.name,
                        'target_ontology': target_ontology.name,
                        'configuration': self.config.matching.__dict__
                    }
                }
                
                self.serializer.serialize_matching_results(results_data, output_path)
                self.logger.info(f"Results saved to {output_path}")
            
            return matching_results
            
        except Exception as e:
            self.logger.error(f"Error during matching: {e}")
            raise
    
    def align_ontologies(self, source_path: str, target_path: str, output_path: str = None):
        try:
            self.logger.info(f"Loading source ontology from {source_path}")
            source_ontology = self.parser.parse_file(source_path)
            
            self.logger.info(f"Loading target ontology from {target_path}")
            target_ontology = self.parser.parse_file(target_path)
            
            self.logger.info("Starting alignment process")
            alignment_results = self.alignment_engine.align_ontologies(source_ontology, target_ontology)
            
            alignment_stats = self.alignment_engine.get_alignment_statistics(alignment_results)
            self.logger.info(f"Alignment completed: {alignment_stats['total_alignments']} alignments found")
            
            if output_path:
                results_data = {
                    'alignments': [
                        {
                            'source_entity': {
                                'id': alignment.source_entity.id,
                                'name': alignment.source_entity.name,
                                'type': alignment.source_entity.entity_type.value
                            },
                            'target_entity': {
                                'id': alignment.target_entity.id,
                                'name': alignment.target_entity.name,
                                'type': alignment.target_entity.entity_type.value
                            },
                            'alignment_type': alignment.alignment_type.value,
                            'confidence': alignment.confidence,
                            'hierarchical_distance': alignment.hierarchical_distance,
                            'structural_similarity': alignment.structural_similarity,
                            'details': alignment.details
                        }
                        for alignment in alignment_results
                    ],
                    'statistics': alignment_stats,
                    'metadata': {
                        'source_ontology': source_ontology.name,
                        'target_ontology': target_ontology.name,
                        'configuration': {
                            'matching': self.config.matching.__dict__,
                            'alignment': self.config.alignment.__dict__
                        }
                    }
                }
                
                self.serializer.serialize_matching_results(results_data, output_path)
                self.logger.info(f"Results saved to {output_path}")
            
            return alignment_results
            
        except Exception as e:
            self.logger.error(f"Error during alignment: {e}")
            raise
    
    def evaluate_results(self, results_path: str, ground_truth_path: str, output_path: str = None):
        try:
            import json
            
            with open(results_path, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            
            with open(ground_truth_path, 'r', encoding='utf-8') as f:
                gt_data = json.load(f)
            
            ground_truth = [
                GroundTruthMapping(
                    source_id=gt['source_id'],
                    target_id=gt['target_id'],
                    is_correct=gt['is_correct'],
                    mapping_type=gt.get('mapping_type', 'equivalence')
                )
                for gt in gt_data.get('mappings', [])
            ]
            
            if 'matches' in results_data:
                from matching.matching_engine import MatchingResult
                from entities.tmf_entity import TMFEntity, EntityType
                
                matching_results = []
                for match_data in results_data['matches']:
                    source_entity = TMFEntity(
                        id=match_data['source_entity']['id'],
                        name=match_data['source_entity']['name'],
                        entity_type=EntityType(match_data['source_entity']['type'])
                    )
                    target_entity = TMFEntity(
                        id=match_data['target_entity']['id'],
                        name=match_data['target_entity']['name'],
                        entity_type=EntityType(match_data['target_entity']['type'])
                    )
                    
                    match = MatchingResult(
                        source_entity=source_entity,
                        target_entity=target_entity,
                        similarity_score=match_data['similarity_score'],
                        confidence=match_data['confidence'],
                        match_type=match_data['match_type'],
                        details=match_data.get('details', {})
                    )
                    matching_results.append(match)
                
                evaluation_results = self.evaluator.evaluate_complete_pipeline(
                    matching_results, [], ground_truth)
            
            elif 'alignments' in results_data:
                from alignment.alignment_engine import AlignmentResult, AlignmentType
                from entities.tmf_entity import TMFEntity, EntityType
                
                alignment_results = []
                for align_data in results_data['alignments']:
                    source_entity = TMFEntity(
                        id=align_data['source_entity']['id'],
                        name=align_data['source_entity']['name'],
                        entity_type=EntityType(align_data['source_entity']['type'])
                    )
                    target_entity = TMFEntity(
                        id=align_data['target_entity']['id'],
                        name=align_data['target_entity']['name'],
                        entity_type=EntityType(align_data['target_entity']['type'])
                    )
                    
                    alignment = AlignmentResult(
                        source_entity=source_entity,
                        target_entity=target_entity,
                        alignment_type=AlignmentType(align_data['alignment_type']),
                        confidence=align_data['confidence'],
                        hierarchical_distance=align_data['hierarchical_distance'],
                        structural_similarity=align_data['structural_similarity'],
                        details=align_data.get('details', {})
                    )
                    alignment_results.append(alignment)
                
                evaluation_results = self.evaluator.evaluate_complete_pipeline(
                    [], alignment_results, ground_truth)
            
            else:
                raise ValueError("Results file must contain either 'matches' or 'alignments'")
            
            report = self.evaluator.generate_evaluation_report(evaluation_results)
            print(report)
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(evaluation_results, f, indent=2, ensure_ascii=False)
                self.logger.info(f"Evaluation results saved to {output_path}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            raise
    
    def export_to_neo4j(self, ontology_path: str, ontology_name: str = None):
        if not self.config.neo4j.enabled:
            raise RuntimeError("Neo4j is not enabled in configuration")
        
        try:
            self.logger.info(f"Loading ontology from {ontology_path}")
            ontology = self.parser.parse_file(ontology_path)
            
            if ontology_name:
                ontology.name = ontology_name
            
            self.logger.info(f"Exporting ontology '{ontology.name}' to Neo4j")
            self.neo4j_exporter.export_ontology(ontology)
            
            self.logger.info(f"Successfully exported ontology to Neo4j")
            
        except Exception as e:
            self.logger.error(f"Error exporting to Neo4j: {e}")
            raise
    
    def import_from_neo4j(self, ontology_name: str, output_path: str = None):
        if not self.config.neo4j.enabled:
            raise RuntimeError("Neo4j is not enabled in configuration")
        
        try:
            self.logger.info(f"Importing ontology '{ontology_name}' from Neo4j")
            ontology = self.neo4j_importer.import_ontology(ontology_name)
            
            if output_path:
                self.serializer.serialize_ontology(ontology, output_path)
                self.logger.info(f"Ontology saved to {output_path}")
            
            return ontology
            
        except Exception as e:
            self.logger.error(f"Error importing from Neo4j: {e}")
            raise
    
    def export_matching_to_neo4j(self, source_path: str, target_path: str, 
                                source_name: str = None, target_name: str = None):
        if not self.config.neo4j.enabled:
            raise RuntimeError("Neo4j is not enabled in configuration")
        
        try:
            matching_results = self.match_ontologies(source_path, target_path)
            
            source_ontology = self.parser.parse_file(source_path)
            target_ontology = self.parser.parse_file(target_path)
            
            source_name = source_name or source_ontology.name
            target_name = target_name or target_ontology.name
            
            self.neo4j_exporter.export_ontology(source_ontology)
            self.neo4j_exporter.export_ontology(target_ontology)
            self.neo4j_exporter.export_matching_results(matching_results, source_name, target_name)
            
            self.logger.info(f"Matching results exported to Neo4j")
            
        except Exception as e:
            self.logger.error(f"Error exporting matching results to Neo4j: {e}")
            raise
    
    def export_alignment_to_neo4j(self, source_path: str, target_path: str,
                                 source_name: str = None, target_name: str = None):
        if not self.config.neo4j.enabled:
            raise RuntimeError("Neo4j is not enabled in configuration")
        
        try:
            alignment_results = self.align_ontologies(source_path, target_path)
            
            source_ontology = self.parser.parse_file(source_path)
            target_ontology = self.parser.parse_file(target_path)
            
            source_name = source_name or source_ontology.name
            target_name = target_name or target_ontology.name
            
            self.neo4j_exporter.export_ontology(source_ontology)
            self.neo4j_exporter.export_ontology(target_ontology)
            self.neo4j_exporter.export_alignment_results(alignment_results, source_name, target_name)
            
            self.logger.info(f"Alignment results exported to Neo4j")
            
        except Exception as e:
            self.logger.error(f"Error exporting alignment results to Neo4j: {e}")
            raise
    
    def convert_to_rdf(self, ontology_path: str, output_path: str, format: str = "turtle"):
        try:
            self.logger.info(f"Loading ontology from {ontology_path}")
            ontology = self.parser.parse_file(ontology_path)
            
            self.logger.info(f"Converting ontology to RDF")
            rdf_graph = self.rdf_converter.ontology_to_rdf(ontology)
            
            self.rdf_converter.save_rdf_to_file(rdf_graph, output_path, format)
            self.logger.info(f"RDF saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error converting to RDF: {e}")
            raise
    
    def convert_from_rdf(self, rdf_path: str, output_path: str = None, ontology_name: str = None):
        try:
            self.logger.info(f"Loading RDF from {rdf_path}")
            rdf_graph = self.rdf_converter.load_rdf_from_file(rdf_path)
            
            self.logger.info(f"Converting RDF to ontology")
            ontology = self.rdf_converter.rdf_to_ontology(rdf_graph, ontology_name)
            
            if output_path:
                self.serializer.serialize_ontology(ontology, output_path)
                self.logger.info(f"Ontology saved to {output_path}")
            
            return ontology
            
        except Exception as e:
            self.logger.error(f"Error converting from RDF: {e}")
            raise
    
    def export_neo4j_to_rdf(self, ontology_name: str, output_path: str, format: str = "turtle"):
        if not self.config.neo4j.enabled:
            raise RuntimeError("Neo4j is not enabled in configuration")
        
        try:
            self.neo4j_rdf_converter.export_ontology_from_neo4j_to_rdf(
                ontology_name, output_path, format)
            
        except Exception as e:
            self.logger.error(f"Error exporting Neo4j to RDF: {e}")
            raise
    
    def import_rdf_to_neo4j(self, rdf_path: str, ontology_name: str):
        if not self.config.neo4j.enabled:
            raise RuntimeError("Neo4j is not enabled in configuration")
        
        try:
            self.neo4j_rdf_converter.import_rdf_to_neo4j(rdf_path, ontology_name)
            
        except Exception as e:
            self.logger.error(f"Error importing RDF to Neo4j: {e}")
            raise
    
    def get_neo4j_statistics(self, ontology_name: str):
        if not self.config.neo4j.enabled:
            raise RuntimeError("Neo4j is not enabled in configuration")
        
        try:
            stats = self.neo4j_importer.get_ontology_statistics(ontology_name)
            print(f"Neo4j Statistics for '{ontology_name}':")
            print(f"  Total entities: {stats['total_entities']}")
            print(f"  Entity types: {', '.join(stats['entity_types'])}")
            print(f"  Relationships:")
            for rel_type, count in stats['relationships'].items():
                print(f"    {rel_type}: {count}")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting Neo4j statistics: {e}")
            raise
    
    def close(self):
        if self.neo4j_connector:
            self.neo4j_connector.close()


def main():
    parser = argparse.ArgumentParser(description="TMF Ontology Matching and Alignment Tool")
    parser.add_argument("command", choices=[
        "match", "align", "evaluate", "create-config",
        "export-neo4j", "import-neo4j", "export-matching-neo4j", "export-alignment-neo4j",
        "to-rdf", "from-rdf", "neo4j-to-rdf", "rdf-to-neo4j", "neo4j-stats"
    ], help="Command to execute")
    parser.add_argument("--source", "-s", help="Source ontology file path")
    parser.add_argument("--target", "-t", help="Target ontology file path")
    parser.add_argument("--output", "-o", help="Output file path")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--ground-truth", "-g", help="Ground truth file path for evaluation")
    parser.add_argument("--results", "-r", help="Results file path for evaluation")
    parser.add_argument("--ontology-name", "-n", help="Ontology name for Neo4j operations")
    parser.add_argument("--source-name", help="Source ontology name for Neo4j")
    parser.add_argument("--target-name", help="Target ontology name for Neo4j")
    parser.add_argument("--format", "-f", default="turtle", help="RDF format (turtle, xml, n3, etc.)")
    
    args = parser.parse_args()
    
    try:
        matcher = TMFOntologyMatcher(args.config)
        
        if args.command == "match":
            if not args.source or not args.target:
                parser.error("match command requires --source and --target arguments")
            matcher.match_ontologies(args.source, args.target, args.output)
        
        elif args.command == "align":
            if not args.source or not args.target:
                parser.error("align command requires --source and --target arguments")
            matcher.align_ontologies(args.source, args.target, args.output)
        
        elif args.command == "evaluate":
            if not args.results or not args.ground_truth:
                parser.error("evaluate command requires --results and --ground-truth arguments")
            matcher.evaluate_results(args.results, args.ground_truth, args.output)
        
        elif args.command == "create-config":
            if not args.output:
                parser.error("create-config command requires --output argument")
            config_manager = ConfigManager()
            config_manager.create_default_config_file(args.output)
            print(f"Default configuration created at {args.output}")
        
        elif args.command == "export-neo4j":
            if not args.source:
                parser.error("export-neo4j command requires --source argument")
            matcher.export_to_neo4j(args.source, args.ontology_name)
        
        elif args.command == "import-neo4j":
            if not args.ontology_name:
                parser.error("import-neo4j command requires --ontology-name argument")
            matcher.import_from_neo4j(args.ontology_name, args.output)
        
        elif args.command == "export-matching-neo4j":
            if not args.source or not args.target:
                parser.error("export-matching-neo4j command requires --source and --target arguments")
            matcher.export_matching_to_neo4j(args.source, args.target, args.source_name, args.target_name)
        
        elif args.command == "export-alignment-neo4j":
            if not args.source or not args.target:
                parser.error("export-alignment-neo4j command requires --source and --target arguments")
            matcher.export_alignment_to_neo4j(args.source, args.target, args.source_name, args.target_name)
        
        elif args.command == "to-rdf":
            if not args.source or not args.output:
                parser.error("to-rdf command requires --source and --output arguments")
            matcher.convert_to_rdf(args.source, args.output, args.format)
        
        elif args.command == "from-rdf":
            if not args.source:
                parser.error("from-rdf command requires --source argument")
            matcher.convert_from_rdf(args.source, args.output, args.ontology_name)
        
        elif args.command == "neo4j-to-rdf":
            if not args.ontology_name or not args.output:
                parser.error("neo4j-to-rdf command requires --ontology-name and --output arguments")
            matcher.export_neo4j_to_rdf(args.ontology_name, args.output, args.format)
        
        elif args.command == "rdf-to-neo4j":
            if not args.source or not args.ontology_name:
                parser.error("rdf-to-neo4j command requires --source and --ontology-name arguments")
            matcher.import_rdf_to_neo4j(args.source, args.ontology_name)
        
        elif args.command == "neo4j-stats":
            if not args.ontology_name:
                parser.error("neo4j-stats command requires --ontology-name argument")
            matcher.get_neo4j_statistics(args.ontology_name)
        
        matcher.close()
        
    except Exception as e:
        logging.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
