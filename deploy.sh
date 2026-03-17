#!/bin/bash

# AWS Deployment Quick Start Script
# This script helps you choose and setup your deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${BLUE}════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}════════════════════════════════════════════${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

# Check prerequisites
print_header "CHECKING PREREQUISITES"

# Check if files exist
if [ ! -f "model.pkl" ]; then
    print_error "model.pkl not found!"
    echo "Please ensure model.pkl exists in current directory"
    exit 1
fi
print_success "model.pkl found"

if [ ! -f "explainer.pkl" ]; then
    print_error "explainer.pkl not found!"
    exit 1
fi
print_success "explainer.pkl found"

if [ ! -f "metadata.json" ]; then
    print_error "metadata.json not found!"
    exit 1
fi
print_success "metadata.json found"

if [ ! -f "app.py" ]; then
    print_error "app.py not found!"
    exit 1
fi
print_success "app.py found"

if [ ! -f "requirements_api.txt" ]; then
    print_error "requirements_api.txt not found!"
    exit 1
fi
print_success "requirements_api.txt found"

# Check git
if ! command -v git &> /dev/null; then
    print_warning "git not found. Please install git for version control."
else
    print_success "git installed"
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 not found!"
    exit 1
fi
print_success "Python 3 installed: $(python3 --version)"

# Check AWS CLI
if ! command -v aws &> /dev/null; then
    print_warning "AWS CLI not found. Run: pip install awscli"
else
    print_success "AWS CLI installed"
fi

# Main menu
print_header "CHOOSE YOUR DEPLOYMENT METHOD"

echo "1) AWS Lambda (Serverless, pay-per-use)"
echo "   Cost: \$0-5/month | Setup: 10 min"
echo ""
echo "2) AWS EC2 (Recommended, full control)"
echo "   Cost: \$11-15/month | Setup: 15 min"
echo ""
echo "3) AWS Elastic Beanstalk (Managed, auto-scaling)"
echo "   Cost: \$15-30/month | Setup: 12 min"
echo ""
echo "4) Docker + ECS (Enterprise, containerized)"
echo "   Cost: \$30-100+/month | Setup: 30 min"
echo ""
echo "5) Test API locally first"
echo ""
echo "6) View deployment guides (no action taken)"
echo ""

read -p "Choose option (1-6): " choice

case $choice in
    1)
        print_header "AWS LAMBDA DEPLOYMENT"
        echo "Follow the guide: AWS_LAMBDA_DEPLOYMENT.md"
        echo ""
        echo "Quick setup:"
        echo "  pip install zappa"
        echo "  zappa init"
        echo "  zappa deploy prod"
        ;;
    2)
        print_header "AWS EC2 DEPLOYMENT"
        echo "Follow the guide: AWS_EC2_DEPLOYMENT.md"
        echo ""
        echo "Prerequisites:"
        echo "  1. Create AWS account (https://aws.amazon.com)"
        echo "  2. Install AWS CLI: pip install awscli"
        echo "  3. Configure: aws configure"
        echo ""
        echo "Then follow AWS_EC2_DEPLOYMENT.md for step-by-step instructions"
        ;;
    3)
        print_header "AWS ELASTIC BEANSTALK DEPLOYMENT"
        echo "Follow the guide: AWS_ELASTIC_BEANSTALK_DEPLOYMENT.md"
        echo ""
        echo "Quick setup:"
        echo "  pip install awsebcli"
        echo "  eb init"
        echo "  eb create"
        echo "  eb deploy"
        ;;
    4)
        print_header "DOCKER + ECS DEPLOYMENT"
        echo "Follow the guide: DOCKER_DEPLOYMENT_GUIDE.md"
        echo ""
        echo "Prerequisites:"
        echo "  1. Install Docker: brew install docker"
        echo "  2. Create AWS ECR repository"
        echo "  3. Follow deployment guide"
        ;;
    5)
        print_header "TESTING API LOCALLY"
        echo "Starting Flask API..."
        
        if [ -d "venv" ]; then
            source venv/bin/activate
        else
            print_warning "No virtual environment found"
            read -p "Create one? (y/n): " create_venv
            if [ "$create_venv" = "y" ]; then
                python3 -m venv venv
                source venv/bin/activate
                pip install -r requirements_api.txt
            fi
        fi
        
        python3 app.py &
        sleep 3
        
        print_header "TESTING ENDPOINTS"
        
        echo "Testing /health endpoint..."
        if curl -s http://localhost:5001/health | python3 -m json.tool > /dev/null 2>&1; then
            print_success "Health check passed!"
        else
            print_error "Health check failed"
        fi
        
        echo ""
        echo "API is running at: http://localhost:5001"
        echo "Press Ctrl+C to stop"
        wait
        ;;
    6)
        print_header "DEPLOYMENT GUIDES"
        echo ""
        echo "📄 Quick Overview:"
        echo "   • DEPLOYMENT_COMPLETE_CHECKLIST.md"
        echo "   • DEPLOYMENT_ARCHITECTURE.md"
        echo ""
        echo "📋 Detailed Guides:"
        echo "   • AWS_LAMBDA_DEPLOYMENT.md"
        echo "   • AWS_EC2_DEPLOYMENT.md"
        echo "   • AWS_ELASTIC_BEANSTALK_DEPLOYMENT.md"
        echo "   • DOCKER_DEPLOYMENT_GUIDE.md"
        echo ""
        echo "🔗 Frontend Integration:"
        echo "   • REACT_INTEGRATION_GUIDE.md"
        echo ""
        echo "📚 General References:"
        echo "   • API_QUICKSTART.md"
        echo "   • README.md"
        ;;
    *)
        print_error "Invalid choice"
        exit 1
        ;;
esac

print_header "NEXT STEPS"

case $choice in
    1|2|3|4)
        echo "1. Review the deployment guide for your chosen option"
        echo "2. Setup AWS account and credentials"
        echo "3. Follow the step-by-step instructions"
        echo "4. Update your React app with the API endpoint"
        echo "5. Deploy your React frontend (Vercel/Netlify)"
        echo ""
        echo "For React integration guide, see: REACT_INTEGRATION_GUIDE.md"
        ;;
    5)
        echo "Local testing complete!"
        echo ""
        echo "Next: Choose a deployment option (run this script again and choose 1-4)"
        ;;
    6)
        echo "Review the guides above to choose your deployment method"
        ;;
esac

echo ""
print_success "Setup complete! Good luck with your deployment! 🚀"
