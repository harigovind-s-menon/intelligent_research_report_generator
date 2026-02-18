#!/bin/bash
# Deploy to AWS ECR
# Usage: ./scripts/deploy-ecr.sh

set -e

# Configuration
AWS_REGION="eu-north-1"
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_BACKEND_REPO="research-report-backend"
ECR_FRONTEND_REPO="research-report-frontend"

echo "=========================================="
echo "AWS Account: $AWS_ACCOUNT_ID"
echo "Region: $AWS_REGION"
echo "=========================================="

# Create ECR repositories if they don't exist
echo "Creating ECR repositories..."
aws ecr describe-repositories --repository-names $ECR_BACKEND_REPO --region $AWS_REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $ECR_BACKEND_REPO --region $AWS_REGION

aws ecr describe-repositories --repository-names $ECR_FRONTEND_REPO --region $AWS_REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $ECR_FRONTEND_REPO --region $AWS_REGION

# Login to ECR
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build and push backend
echo "Building backend image..."
docker build -t $ECR_BACKEND_REPO:latest -f Dockerfile .

echo "Tagging and pushing backend..."
docker tag $ECR_BACKEND_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_BACKEND_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_BACKEND_REPO:latest

# Build and push frontend
echo "Building frontend image..."
docker build -t $ECR_FRONTEND_REPO:latest -f Dockerfile.frontend .

echo "Tagging and pushing frontend..."
docker tag $ECR_FRONTEND_REPO:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_FRONTEND_REPO:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_FRONTEND_REPO:latest

echo "=========================================="
echo "Done! Images pushed to ECR:"
echo "  Backend:  $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_BACKEND_REPO:latest"
echo "  Frontend: $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_FRONTEND_REPO:latest"
echo "=========================================="
