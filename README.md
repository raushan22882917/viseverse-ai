# VisaVerse Guardian AI

AI-driven global mobility intelligence platform that provides predictive, explainable, and memory-driven intelligence for global mobility decisions.

## Overview

VisaVerse Guardian AI is a production-ready, agent-based AI system that:

- Predicts visa and compliance success probability
- Validates documents using AI and rules
- Explains risks with transparent reasoning
- Learns persistently from past cases
- Supports global, multilingual documents

## Architecture

The system follows a microservices architecture with the following components:

- **API Gateway**: FastAPI-based orchestration layer
- **Document Processing Service**: PaddleOCR-based multilingual document extraction
- **Graph Reasoning Service**: Neo4j-based rule validation and compliance checking
- **Risk Assessment Service**: Gemini LLM-based probability calculation and explanation
- **Memory Service**: MemMachine-based persistent learning and historical analysis

## Quick Start

### Prerequisites

- Python 3.9+
- Docker (for containerized deployment)
- Google Cloud Platform account (for production deployment)

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/raushan22882917/viseverse-ai.git
cd viseverse-ai
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy environment configuration:
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. Run the application:
```bash
python -m uvicorn src.visaverse.api.main:app --reload
```

6. Access the API documentation:
```
http://localhost:8000/docs
```

### Testing

Run unit tests:
```bash
pytest tests/
```

Run property-based tests:
```bash
pytest tests/ -k "property"
```

### Docker Deployment

1. Build the container:
```bash
docker build -t visaverse-guardian-ai .
```

2. Run the container:
```bash
docker run -p 8000:8000 --env-file .env visaverse-guardian-ai
```

### Google Cloud Platform Deployment

1. Set up GCP project and enable required APIs:
   - Cloud Run API
   - Cloud Build API
   - Container Registry API
   - Cloud Storage API

2. Deploy using Cloud Build:
```bash
gcloud builds submit --config cloudbuild.yaml
```

## Project Structure

```
visaverse-guardian-ai/
├── src/
│   └── visaverse/
│       ├── core/                 # Core models and interfaces
│       │   ├── models.py         # Pydantic data models
│       │   └── interfaces.py     # Abstract service interfaces
│       ├── services/             # Service implementations
│       │   ├── document/         # Document processing service
│       │   ├── graph/            # Graph reasoning service
│       │   ├── risk/             # Risk assessment service
│       │   └── memory/           # Memory service
│       └── api/                  # FastAPI application
│           └── main.py           # Main application entry point
├── tests/                        # Test suite
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
├── Dockerfile                   # Container configuration
├── cloudbuild.yaml             # GCP Cloud Build configuration
└── README.md                   # This file
```

## Development Guidelines

### Code Style

- Use Black for code formatting
- Use isort for import sorting
- Follow PEP 8 guidelines
- Use type hints throughout

### Testing

- Write unit tests for all new functionality
- Use property-based testing with Hypothesis for core logic
- Maintain test coverage above 80%
- Test both success and error scenarios

### Documentation

- Document all public APIs
- Include docstrings for all functions and classes
- Update README for significant changes
- Maintain API documentation in FastAPI

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support, please contact the development team.
