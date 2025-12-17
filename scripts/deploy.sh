#!/bin/bash
# Deployment script for VisaVerse Guardian AI

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${GCP_PROJECT_ID:-""}
REGION=${GCP_REGION:-"us-central1"}
SERVICE_NAME="visaverse-guardian-ai"

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

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if gcloud is installed
    if ! command -v gcloud &> /dev/null; then
        log_error "gcloud CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if docker is installed
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if terraform is installed
    if ! command -v terraform &> /dev/null; then
        log_warn "Terraform is not installed. Infrastructure deployment will be skipped."
    fi
    
    # Check if project ID is set
    if [ -z "$PROJECT_ID" ]; then
        log_error "GCP_PROJECT_ID environment variable is not set."
        echo "Please set it with: export GCP_PROJECT_ID=your-project-id"
        exit 1
    fi
    
    log_info "Prerequisites check completed."
}

authenticate_gcloud() {
    log_info "Authenticating with Google Cloud..."
    
    # Check if already authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        log_info "No active authentication found. Please authenticate:"
        gcloud auth login
    fi
    
    # Set project
    gcloud config set project $PROJECT_ID
    
    # Enable required APIs
    log_info "Enabling required Google Cloud APIs..."
    gcloud services enable \
        run.googleapis.com \
        cloudbuild.googleapis.com \
        containerregistry.googleapis.com \
        secretmanager.googleapis.com \
        storage.googleapis.com \
        aiplatform.googleapis.com \
        monitoring.googleapis.com \
        logging.googleapis.com \
        cloudtrace.googleapis.com \
        compute.googleapis.com
    
    log_info "Google Cloud authentication completed."
}

build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Get current commit SHA
    COMMIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "latest")
    IMAGE_TAG="gcr.io/$PROJECT_ID/$SERVICE_NAME:$COMMIT_SHA"
    
    # Build image
    log_info "Building Docker image: $IMAGE_TAG"
    docker build -t $IMAGE_TAG .
    
    # Configure Docker for GCR
    gcloud auth configure-docker --quiet
    
    # Push image
    log_info "Pushing Docker image to Google Container Registry..."
    docker push $IMAGE_TAG
    
    log_info "Docker image build and push completed."
    echo "IMAGE_TAG=$IMAGE_TAG"
}

deploy_infrastructure() {
    if ! command -v terraform &> /dev/null; then
        log_warn "Terraform not found. Skipping infrastructure deployment."
        return 0
    fi
    
    log_info "Deploying infrastructure with Terraform..."
    
    cd terraform
    
    # Check if terraform.tfvars exists
    if [ ! -f "terraform.tfvars" ]; then
        log_error "terraform.tfvars not found. Please create it from terraform.tfvars.example"
        exit 1
    fi
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    log_info "Planning Terraform deployment..."
    terraform plan -var="project_id=$PROJECT_ID"
    
    # Ask for confirmation
    read -p "Do you want to apply these changes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Apply deployment
        log_info "Applying Terraform deployment..."
        terraform apply -var="project_id=$PROJECT_ID" -auto-approve
        
        log_info "Infrastructure deployment completed."
    else
        log_info "Infrastructure deployment cancelled."
    fi
    
    cd ..
}

deploy_application() {
    local DEPLOYMENT_TYPE=${1:-"cloudrun"}
    
    case $DEPLOYMENT_TYPE in
        "cloudrun")
            deploy_to_cloudrun
            ;;
        "kubernetes")
            deploy_to_kubernetes
            ;;
        *)
            log_error "Unknown deployment type: $DEPLOYMENT_TYPE"
            exit 1
            ;;
    esac
}

deploy_to_cloudrun() {
    log_info "Deploying application to Cloud Run..."
    
    # Get current commit SHA
    COMMIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "latest")
    IMAGE_TAG="gcr.io/$PROJECT_ID/$SERVICE_NAME:$COMMIT_SHA"
    
    # Create secrets in Secret Manager if they don't exist
    create_secrets_if_needed
    
    # Deploy to Cloud Run
    gcloud run deploy $SERVICE_NAME \
        --image $IMAGE_TAG \
        --region $REGION \
        --platform managed \
        --port 8080 \
        --memory 2Gi \
        --cpu 2 \
        --min-instances 1 \
        --max-instances 10 \
        --concurrency 80 \
        --timeout 300 \
        --set-env-vars "GCP_PROJECT_ID=$PROJECT_ID,ENVIRONMENT=cloudrun,LOG_LEVEL=INFO,GCP_REGION=$REGION" \
        --set-secrets "GEMINI_API_KEY=gemini-api-key:latest,NEO4J_PASSWORD=neo4j-password:latest,GOOGLE_API_KEY=google-api-key:latest" \
        --allow-unauthenticated
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)")
    
    log_info "Cloud Run deployment completed."
    log_info "Service URL: $SERVICE_URL"
}

deploy_to_kubernetes() {
    log_info "Deploying application to Kubernetes..."
    
    # Check if kubectl is available
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is not installed. Please install it first."
        exit 1
    fi
    
    # Get current commit SHA
    COMMIT_SHA=$(git rev-parse HEAD 2>/dev/null || echo "latest")
    IMAGE_TAG="gcr.io/$PROJECT_ID/$SERVICE_NAME:$COMMIT_SHA"
    
    # Update deployment YAML with correct image and project ID
    sed -i.bak "s|gcr.io/PROJECT_ID/visaverse-guardian-ai:latest|$IMAGE_TAG|g" k8s/deployment.yaml
    sed -i.bak "s|PROJECT_ID|$PROJECT_ID|g" k8s/deployment.yaml
    
    # Create namespace if it doesn't exist
    kubectl create namespace visaverse --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    create_k8s_secrets
    
    # Apply deployment
    kubectl apply -f k8s/deployment.yaml -n visaverse
    
    # Wait for deployment to be ready
    kubectl rollout status deployment/visaverse-guardian-ai -n visaverse --timeout=300s
    
    # Get service URL (if using ingress)
    log_info "Kubernetes deployment completed."
    log_info "Check service status with: kubectl get pods -n visaverse"
    
    # Restore original deployment file
    mv k8s/deployment.yaml.bak k8s/deployment.yaml
}

create_secrets_if_needed() {
    log_info "Creating secrets in Secret Manager..."
    
    # List of secrets to create
    declare -A secrets=(
        ["gemini-api-key"]="your-gemini-api-key-here"
        ["neo4j-password"]="your-neo4j-password-here"
        ["google-api-key"]="your-google-api-key-here"
    )
    
    for secret_name in "${!secrets[@]}"; do
        if ! gcloud secrets describe $secret_name &>/dev/null; then
            log_info "Creating secret: $secret_name"
            echo -n "${secrets[$secret_name]}" | gcloud secrets create $secret_name --data-file=-
            log_warn "Please update the secret '$secret_name' with the actual value:"
            log_warn "echo -n 'actual-secret-value' | gcloud secrets versions add $secret_name --data-file=-"
        else
            log_info "Secret $secret_name already exists"
        fi
    done
}

create_k8s_secrets() {
    log_info "Creating Kubernetes secrets..."
    
    # Create secret for API keys (you should replace these with actual values)
    kubectl create secret generic visaverse-secrets \
        --from-literal=gemini-api-key="your-gemini-api-key" \
        --from-literal=neo4j-password="your-neo4j-password" \
        --from-literal=google-api-key="your-google-api-key" \
        --namespace=visaverse \
        --dry-run=client -o yaml | kubectl apply -f -
    
    log_warn "Please update the Kubernetes secrets with actual values:"
    log_warn "kubectl patch secret visaverse-secrets -n visaverse -p='{\"data\":{\"gemini-api-key\":\"<base64-encoded-key>\"}}'"
}

run_health_check() {
    log_info "Running health check..."
    
    # Get service URL
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)" 2>/dev/null || echo "")
    
    if [ -z "$SERVICE_URL" ]; then
        log_error "Could not get service URL. Health check skipped."
        return 1
    fi
    
    # Wait a bit for service to be ready
    sleep 10
    
    # Check health endpoint
    log_info "Checking health endpoint: $SERVICE_URL/health"
    
    if curl -f -s "$SERVICE_URL/health" > /dev/null; then
        log_info "Health check passed!"
        
        # Show detailed health
        log_info "Detailed health status:"
        curl -s "$SERVICE_URL/health/detailed" | python3 -m json.tool || echo "Could not parse health response"
    else
        log_error "Health check failed!"
        return 1
    fi
}

setup_monitoring() {
    log_info "Setting up monitoring and alerting..."
    
    # Create notification channel (email)
    read -p "Enter email for alerts (or press Enter to skip): " ALERT_EMAIL
    
    if [ ! -z "$ALERT_EMAIL" ]; then
        log_info "Creating notification channel for $ALERT_EMAIL"
        
        # This would typically be done through Terraform or gcloud commands
        # For now, we'll just log the instruction
        log_info "Please manually set up notification channels in the Google Cloud Console:"
        log_info "https://console.cloud.google.com/monitoring/alerting/notifications"
    fi
    
    log_info "Monitoring setup completed."
}

cleanup_old_images() {
    log_info "Cleaning up old Docker images..."
    
    # Keep only the last 5 images
    gcloud container images list-tags gcr.io/$PROJECT_ID/$SERVICE_NAME \
        --limit=999999 \
        --sort-by=TIMESTAMP \
        --format="get(digest)" | tail -n +6 | \
        xargs -I {} gcloud container images delete gcr.io/$PROJECT_ID/$SERVICE_NAME@{} --quiet || true
    
    log_info "Image cleanup completed."
}

main() {
    log_info "Starting VisaVerse Guardian AI deployment..."
    
    # Parse command line arguments
    SKIP_INFRA=false
    SKIP_BUILD=false
    SKIP_DEPLOY=false
    DEPLOYMENT_TYPE="cloudrun"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-infra)
                SKIP_INFRA=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-deploy)
                SKIP_DEPLOY=true
                shift
                ;;
            --type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            --help)
                echo "Usage: $0 [OPTIONS]"
                echo "Options:"
                echo "  --type TYPE    Deployment type: cloudrun (default) or kubernetes"
                echo "  --skip-infra   Skip infrastructure deployment"
                echo "  --skip-build   Skip Docker image build"
                echo "  --skip-deploy  Skip application deployment"
                echo "  --help         Show this help message"
                echo ""
                echo "Examples:"
                echo "  $0                           # Deploy to Cloud Run"
                echo "  $0 --type kubernetes         # Deploy to Kubernetes"
                echo "  $0 --skip-infra --skip-build # Only deploy application"
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done
    
    # Run deployment steps
    check_prerequisites
    authenticate_gcloud
    
    if [ "$SKIP_INFRA" = false ]; then
        deploy_infrastructure
    fi
    
    if [ "$SKIP_BUILD" = false ]; then
        build_and_push_image
    fi
    
    if [ "$SKIP_DEPLOY" = false ]; then
        deploy_application $DEPLOYMENT_TYPE
        if [ "$DEPLOYMENT_TYPE" = "cloudrun" ]; then
            run_health_check
            setup_monitoring
            cleanup_old_images
        fi
    fi
    
    log_info "Deployment completed successfully!"
    
    # Show final information
    SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)" 2>/dev/null || echo "")
    if [ ! -z "$SERVICE_URL" ]; then
        echo
        log_info "=== Deployment Summary ==="
        log_info "Service URL: $SERVICE_URL"
        log_info "Health Check: $SERVICE_URL/health"
        log_info "API Documentation: $SERVICE_URL/docs"
        log_info "Project: $PROJECT_ID"
        log_info "Region: $REGION"
        echo
    fi
}

# Run main function
main "$@"