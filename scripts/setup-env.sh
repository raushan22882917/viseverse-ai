#!/bin/bash
# Environment setup script for VisaVerse Guardian AI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_prompt() {
    echo -e "${BLUE}[PROMPT]${NC} $1"
}

show_help() {
    echo "Usage: $0 [ENVIRONMENT]"
    echo ""
    echo "ENVIRONMENT can be:"
    echo "  local      - Set up for local development"
    echo "  kubernetes - Set up for Kubernetes deployment"
    echo "  cloudrun   - Set up for Cloud Run deployment"
    echo ""
    echo "This script will:"
    echo "  1. Copy the appropriate .env template"
    echo "  2. Help you configure the necessary values"
    echo "  3. Validate the configuration"
    echo ""
    echo "Examples:"
    echo "  $0 local      # Set up for local development"
    echo "  $0 kubernetes # Set up for Kubernetes deployment"
    echo "  $0 cloudrun   # Set up for Cloud Run deployment"
}

setup_local_env() {
    log_info "Setting up local development environment..."
    
    # Copy local environment template
    cp .env.local .env
    
    log_info "Local environment template copied to .env"
    log_warn "Please update the following values in .env:"
    echo "  - NEO4J_PASSWORD: Set your local Neo4j password"
    echo "  - GOOGLE_API_KEY: Set your Google API key"
    echo "  - GEMINI_API_KEY: Set your Gemini API key"
    echo "  - GCP_PROJECT_ID: Set your GCP project ID"
    
    # Offer to start local services
    log_prompt "Would you like to start local services with Docker Compose? (y/N)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        if [ -f "docker-compose.yml" ]; then
            docker-compose up -d
            log_info "Local services started with Docker Compose"
        else
            log_warn "docker-compose.yml not found. You'll need to start services manually."
        fi
    fi
}

setup_kubernetes_env() {
    log_info "Setting up Kubernetes deployment environment..."
    
    # Copy Kubernetes environment template
    cp .env.kubernetes .env.k8s
    
    log_info "Kubernetes environment template copied to .env.k8s"
    log_warn "For Kubernetes deployment, secrets are managed through Kubernetes secrets."
    log_info "The deployment script will create the necessary secrets."
    
    # Check if kubectl is available
    if command -v kubectl &> /dev/null; then
        log_info "kubectl is available"
        
        # Check current context
        current_context=$(kubectl config current-context 2>/dev/null || echo "none")
        log_info "Current kubectl context: $current_context"
        
        if [ "$current_context" = "none" ]; then
            log_warn "No kubectl context set. Please configure kubectl to connect to your cluster."
        fi
    else
        log_warn "kubectl not found. Please install kubectl for Kubernetes deployment."
    fi
}

setup_cloudrun_env() {
    log_info "Setting up Cloud Run deployment environment..."
    
    # Copy Cloud Run environment template
    cp .env.cloudrun .env.cloudrun.configured
    
    log_info "Cloud Run environment template copied to .env.cloudrun.configured"
    
    # Get GCP project ID
    log_prompt "Enter your GCP Project ID:"
    read -r project_id
    
    if [ -n "$project_id" ]; then
        sed -i.bak "s/\${GCP_PROJECT_ID}/$project_id/g" .env.cloudrun.configured
        log_info "GCP Project ID set to: $project_id"
    fi
    
    # Get region
    log_prompt "Enter your preferred GCP region (default: us-central1):"
    read -r region
    region=${region:-"us-central1"}
    
    sed -i.bak "s/us-central1/$region/g" .env.cloudrun.configured
    log_info "GCP Region set to: $region"
    
    # Clean up backup file
    rm -f .env.cloudrun.configured.bak
    
    log_warn "For Cloud Run deployment, secrets are managed through Google Secret Manager."
    log_info "The deployment script will create the necessary secrets."
    
    # Check if gcloud is available
    if command -v gcloud &> /dev/null; then
        log_info "gcloud CLI is available"
        
        # Check current project
        current_project=$(gcloud config get-value project 2>/dev/null || echo "none")
        log_info "Current gcloud project: $current_project"
        
        if [ "$current_project" != "$project_id" ]; then
            log_prompt "Would you like to set the gcloud project to $project_id? (y/N)"
            read -r response
            if [[ "$response" =~ ^[Yy]$ ]]; then
                gcloud config set project "$project_id"
                log_info "gcloud project set to: $project_id"
            fi
        fi
    else
        log_warn "gcloud CLI not found. Please install gcloud for Cloud Run deployment."
    fi
}

validate_environment() {
    local env_type=$1
    
    log_info "Validating environment configuration..."
    
    case $env_type in
        "local")
            if [ -f ".env" ]; then
                log_info "Local .env file exists"
                
                # Check for required variables
                if grep -q "NEO4J_PASSWORD=your-neo4j-password" .env; then
                    log_warn "Please update NEO4J_PASSWORD in .env"
                fi
                
                if grep -q "GOOGLE_API_KEY=your-dev-google-api-key" .env; then
                    log_warn "Please update GOOGLE_API_KEY in .env"
                fi
                
                if grep -q "GEMINI_API_KEY=your-dev-gemini-api-key" .env; then
                    log_warn "Please update GEMINI_API_KEY in .env"
                fi
            else
                log_error ".env file not found"
                return 1
            fi
            ;;
        "kubernetes")
            if [ -f ".env.k8s" ]; then
                log_info "Kubernetes environment file exists"
            else
                log_error ".env.k8s file not found"
                return 1
            fi
            ;;
        "cloudrun")
            if [ -f ".env.cloudrun.configured" ]; then
                log_info "Cloud Run environment file exists"
            else
                log_error ".env.cloudrun.configured file not found"
                return 1
            fi
            ;;
    esac
    
    log_info "Environment validation completed"
}

create_docker_compose() {
    log_info "Creating docker-compose.yml for local development..."
    
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  neo4j:
    image: neo4j:5.15
    container_name: visaverse-neo4j
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/local-dev-password
      - NEO4J_PLUGINS=["apoc"]
      - NEO4J_dbms_security_procedures_unrestricted=apoc.*
    volumes:
      - neo4j_data:/data
      - neo4j_logs:/logs
      - neo4j_import:/var/lib/neo4j/import
      - neo4j_plugins:/plugins
    networks:
      - visaverse-network

  redis:
    image: redis:7-alpine
    container_name: visaverse-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    networks:
      - visaverse-network

  # Placeholder for document service
  document-service:
    image: nginx:alpine
    container_name: visaverse-document-service
    ports:
      - "8001:80"
    networks:
      - visaverse-network

  # Placeholder for graph service
  graph-service:
    image: nginx:alpine
    container_name: visaverse-graph-service
    ports:
      - "8002:80"
    networks:
      - visaverse-network

  # Placeholder for risk service
  risk-service:
    image: nginx:alpine
    container_name: visaverse-risk-service
    ports:
      - "8003:80"
    networks:
      - visaverse-network

  # Placeholder for memory service
  memory-service:
    image: nginx:alpine
    container_name: visaverse-memory-service
    ports:
      - "8004:80"
    networks:
      - visaverse-network

volumes:
  neo4j_data:
  neo4j_logs:
  neo4j_import:
  neo4j_plugins:
  redis_data:

networks:
  visaverse-network:
    driver: bridge
EOF

    log_info "docker-compose.yml created for local development"
    log_info "Start services with: docker-compose up -d"
    log_info "Stop services with: docker-compose down"
}

main() {
    local environment=${1:-""}
    
    if [ -z "$environment" ]; then
        log_error "Please specify an environment"
        show_help
        exit 1
    fi
    
    case $environment in
        "local")
            setup_local_env
            if [ ! -f "docker-compose.yml" ]; then
                log_prompt "Would you like to create a docker-compose.yml for local services? (y/N)"
                read -r response
                if [[ "$response" =~ ^[Yy]$ ]]; then
                    create_docker_compose
                fi
            fi
            validate_environment "local"
            ;;
        "kubernetes")
            setup_kubernetes_env
            validate_environment "kubernetes"
            ;;
        "cloudrun")
            setup_cloudrun_env
            validate_environment "cloudrun"
            ;;
        "help"|"--help"|"-h")
            show_help
            exit 0
            ;;
        *)
            log_error "Unknown environment: $environment"
            show_help
            exit 1
            ;;
    esac
    
    log_info "Environment setup completed for: $environment"
    
    # Show next steps
    echo ""
    log_info "=== Next Steps ==="
    case $environment in
        "local")
            echo "1. Update the values in .env file"
            echo "2. Start local services: docker-compose up -d"
            echo "3. Run the application: python -m uvicorn src.visaverse.api.main:app --reload"
            ;;
        "kubernetes")
            echo "1. Configure kubectl to connect to your cluster"
            echo "2. Run deployment: ./scripts/deploy.sh --type kubernetes"
            ;;
        "cloudrun")
            echo "1. Authenticate with gcloud: gcloud auth login"
            echo "2. Run deployment: ./scripts/deploy.sh --type cloudrun"
            ;;
    esac
    echo ""
}

# Run main function
main "$@"