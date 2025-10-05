from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import json
import yaml
from pathlib import Path
import logging


@dataclass
class ComparisonStrategyConfig:
    strategy_type: str
    weight: float = 1.0
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MatchingConfig:
    similarity_threshold: float = 0.6
    confidence_threshold: float = 0.5
    max_matches_per_entity: int = 5
    enable_parallel_processing: bool = True
    max_workers: int = 4
    enable_context_matching: bool = True
    context_depth: int = 2
    filter_by_entity_type: bool = True
    enable_bidirectional_matching: bool = False
    comparison_strategies: List[ComparisonStrategyConfig] = field(default_factory=list)


@dataclass
class AlignmentConfig:
    equivalence_threshold: float = 0.9
    subsumption_threshold: float = 0.7
    overlap_threshold: float = 0.5
    enable_hierarchical_analysis: bool = True
    max_hierarchical_depth: int = 5
    consider_sibling_relationships: bool = True
    enable_structural_propagation: bool = True
    confidence_decay_factor: float = 0.1


@dataclass
class ParsingConfig:
    default_namespace: str = "http://www.tmforum.org/ontology#"
    enable_validation: bool = True
    strict_parsing: bool = False
    auto_generate_ids: bool = True
    preserve_original_structure: bool = True


@dataclass
class SerializationConfig:
    default_format: str = "json"
    pretty_print: bool = True
    include_metadata: bool = True
    include_timestamps: bool = True
    compression_enabled: bool = False


@dataclass
class LoggingConfig:
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10485760
    backup_count: int = 5


@dataclass
class TMFConfig:
    matching: MatchingConfig = field(default_factory=MatchingConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    parsing: ParsingConfig = field(default_factory=ParsingConfig)
    serialization: SerializationConfig = field(default_factory=SerializationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TMFConfig':
        config = cls()
        
        if 'matching' in data:
            matching_data = data['matching']
            config.matching = MatchingConfig(**matching_data)
        
        if 'alignment' in data:
            alignment_data = data['alignment']
            config.alignment = AlignmentConfig(**alignment_data)
        
        if 'parsing' in data:
            parsing_data = data['parsing']
            config.parsing = ParsingConfig(**parsing_data)
        
        if 'serialization' in data:
            serialization_data = data['serialization']
            config.serialization = SerializationConfig(**serialization_data)
        
        if 'logging' in data:
            logging_data = data['logging']
            config.logging = LoggingConfig(**logging_data)
        
        return config


class ConfigManager:
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        self._config = None
    
    def load_config(self, config_path: Optional[str] = None) -> TMFConfig:
        path = config_path or self.config_path
        
        if not path or not Path(path).exists():
            self.logger.info("No config file found, using default configuration")
            self._config = TMFConfig()
            return self._config
        
        try:
            with open(path, 'r', encoding='utf-8') as file:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    data = yaml.safe_load(file)
                else:
                    data = json.load(file)
            
            self._config = TMFConfig.from_dict(data)
            self.logger.info(f"Configuration loaded from {path}")
            return self._config
            
        except Exception as e:
            self.logger.error(f"Error loading configuration from {path}: {e}")
            self.logger.info("Using default configuration")
            self._config = TMFConfig()
            return self._config
    
    def save_config(self, config: TMFConfig, config_path: Optional[str] = None) -> None:
        path = config_path or self.config_path
        
        if not path:
            raise ValueError("No config path specified")
        
        try:
            data = config.to_dict()
            
            with open(path, 'w', encoding='utf-8') as file:
                if path.endswith('.yaml') or path.endswith('.yml'):
                    yaml.dump(data, file, default_flow_style=False, indent=2)
                else:
                    json.dump(data, file, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Configuration saved to {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving configuration to {path}: {e}")
            raise
    
    def get_config(self) -> TMFConfig:
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        if self._config is None:
            self._config = TMFConfig()
        
        for key, value in updates.items():
            if hasattr(self._config, key):
                if isinstance(getattr(self._config, key), dict):
                    getattr(self._config, key).update(value)
                else:
                    setattr(self._config, key, value)
    
    def create_default_config_file(self, config_path: str) -> None:
        default_config = TMFConfig()
        
        default_config.matching.comparison_strategies = [
            ComparisonStrategyConfig(
                strategy_type="ExactMatchStrategy",
                weight=1.0,
                parameters={"case_sensitive": False}
            ),
            ComparisonStrategyConfig(
                strategy_type="SynonymMatchStrategy",
                weight=0.9,
                parameters={"synonym_threshold": 0.8}
            ),
            ComparisonStrategyConfig(
                strategy_type="StructuralSimilarityStrategy",
                weight=0.7,
                parameters={}
            ),
            ComparisonStrategyConfig(
                strategy_type="SemanticSimilarityStrategy",
                weight=0.8,
                parameters={}
            )
        ]
        
        self.save_config(default_config, config_path)


class ConfigValidator:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_config(self, config: TMFConfig) -> List[str]:
        errors = []
        
        errors.extend(self._validate_matching_config(config.matching))
        errors.extend(self._validate_alignment_config(config.alignment))
        errors.extend(self._validate_parsing_config(config.parsing))
        errors.extend(self._validate_serialization_config(config.serialization))
        errors.extend(self._validate_logging_config(config.logging))
        
        return errors
    
    def _validate_matching_config(self, config: MatchingConfig) -> List[str]:
        errors = []
        
        if not 0.0 <= config.similarity_threshold <= 1.0:
            errors.append("similarity_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= config.confidence_threshold <= 1.0:
            errors.append("confidence_threshold must be between 0.0 and 1.0")
        
        if config.max_matches_per_entity < 1:
            errors.append("max_matches_per_entity must be at least 1")
        
        if config.max_workers < 1:
            errors.append("max_workers must be at least 1")
        
        if config.context_depth < 0:
            errors.append("context_depth must be non-negative")
        
        return errors
    
    def _validate_alignment_config(self, config: AlignmentConfig) -> List[str]:
        errors = []
        
        if not 0.0 <= config.equivalence_threshold <= 1.0:
            errors.append("equivalence_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= config.subsumption_threshold <= 1.0:
            errors.append("subsumption_threshold must be between 0.0 and 1.0")
        
        if not 0.0 <= config.overlap_threshold <= 1.0:
            errors.append("overlap_threshold must be between 0.0 and 1.0")
        
        if config.equivalence_threshold < config.subsumption_threshold:
            errors.append("equivalence_threshold should be >= subsumption_threshold")
        
        if config.subsumption_threshold < config.overlap_threshold:
            errors.append("subsumption_threshold should be >= overlap_threshold")
        
        if config.max_hierarchical_depth < 1:
            errors.append("max_hierarchical_depth must be at least 1")
        
        if not 0.0 <= config.confidence_decay_factor <= 1.0:
            errors.append("confidence_decay_factor must be between 0.0 and 1.0")
        
        return errors
    
    def _validate_parsing_config(self, config: ParsingConfig) -> List[str]:
        errors = []
        
        if not config.default_namespace:
            errors.append("default_namespace cannot be empty")
        
        return errors
    
    def _validate_serialization_config(self, config: SerializationConfig) -> List[str]:
        errors = []
        
        valid_formats = ["json", "xml", "yaml"]
        if config.default_format not in valid_formats:
            errors.append(f"default_format must be one of {valid_formats}")
        
        return errors
    
    def _validate_logging_config(self, config: LoggingConfig) -> List[str]:
        errors = []
        
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if config.level not in valid_levels:
            errors.append(f"logging level must be one of {valid_levels}")
        
        if config.max_file_size < 1024:
            errors.append("max_file_size must be at least 1024 bytes")
        
        if config.backup_count < 0:
            errors.append("backup_count must be non-negative")
        
        return errors


class EnvironmentConfigLoader:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def load_from_environment(self) -> Dict[str, Any]:
        import os
        
        config_updates = {}
        
        if os.getenv('TMF_SIMILARITY_THRESHOLD'):
            try:
                threshold = float(os.getenv('TMF_SIMILARITY_THRESHOLD'))
                config_updates['matching'] = {'similarity_threshold': threshold}
            except ValueError:
                self.logger.warning("Invalid TMF_SIMILARITY_THRESHOLD value")
        
        if os.getenv('TMF_CONFIDENCE_THRESHOLD'):
            try:
                threshold = float(os.getenv('TMF_CONFIDENCE_THRESHOLD'))
                if 'matching' not in config_updates:
                    config_updates['matching'] = {}
                config_updates['matching']['confidence_threshold'] = threshold
            except ValueError:
                self.logger.warning("Invalid TMF_CONFIDENCE_THRESHOLD value")
        
        if os.getenv('TMF_MAX_WORKERS'):
            try:
                workers = int(os.getenv('TMF_MAX_WORKERS'))
                if 'matching' not in config_updates:
                    config_updates['matching'] = {}
                config_updates['matching']['max_workers'] = workers
            except ValueError:
                self.logger.warning("Invalid TMF_MAX_WORKERS value")
        
        if os.getenv('TMF_LOG_LEVEL'):
            level = os.getenv('TMF_LOG_LEVEL').upper()
            if level in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']:
                config_updates['logging'] = {'level': level}
            else:
                self.logger.warning("Invalid TMF_LOG_LEVEL value")
        
        if os.getenv('TMF_LOG_FILE'):
            if 'logging' not in config_updates:
                config_updates['logging'] = {}
            config_updates['logging']['file_path'] = os.getenv('TMF_LOG_FILE')
        
        return config_updates


def setup_logging(config: LoggingConfig) -> None:
    import logging.handlers
    
    level = getattr(logging, config.level.upper())
    
    formatter = logging.Formatter(config.format)
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    if config.file_path:
        file_handler = logging.handlers.RotatingFileHandler(
            config.file_path,
            maxBytes=config.max_file_size,
            backupCount=config.backup_count
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
