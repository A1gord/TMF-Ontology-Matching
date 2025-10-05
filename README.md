# TMF Ontology Matching and Alignment Framework

Framework for ontology matching and alignment in TMF knowledge graphs with Neo4j and RDF support.

## Features

- **Ontology Matching**: Semantic similarity-based entity matching with multiple comparison strategies
- **Ontology Alignment**: Hierarchical relationship-aware alignment with structural propagation
- **Neo4j Integration**: Export/import ontologies and results to/from Neo4j graph database
- **RDF Support**: Convert between ontologies and RDF formats (Turtle, RDF/XML, N3, JSON-LD)
- **Multiple Formats**: Parse XML/OWL/RDF and custom TMF format files
- **Quality Metrics**: Precision, recall, F1-score evaluation with ranking metrics
- **Configurable**: YAML/JSON configuration with environment variable support

## Project Structure

```
src/
├── entities/           # TMF entity models and knowledge graph
│   ├── tmf_entity.py   # Core TMF entity classes
│   └── knowledge_graph.py  # Knowledge graph implementation
├── matching/           # Ontology matching algorithms
│   ├── comparison_strategies.py  # Matching strategies (exact, semantic, structural)
│   └── matching_engine.py       # Main matching engine
├── alignment/          # Ontology alignment algorithms  
│   └── alignment_engine.py      # Hierarchical alignment engine
└── utils/              # Utilities and I/O
    ├── xml_parser.py   # XML/OWL/TMF format parsers
    ├── serialization.py # JSON/XML serialization
    ├── neo4j_handler.py # Neo4j database integration
    ├── rdf_converter.py # RDF format conversion
    ├── metrics.py      # Quality evaluation metrics
    └── config.py       # Configuration management
```