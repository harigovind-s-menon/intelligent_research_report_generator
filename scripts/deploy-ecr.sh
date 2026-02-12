#!/bin/bash
# Quick AWS Deployment Script
# This script automates the basic deployment steps

set -e

# Configuration - UPDATE THESE
export AWS_REGION="eu-north-1"
export PROJECT_NAME="research-report"
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

echo "=== AWS Account: $AWS_ACCOUNT_ID ==="
echo "=== Region: $AWS_REGION ==="

# Step 1: Create ECR repositories
echo "Creating ECR repositories..."
aws ecr create-repository --repository-name ${PROJECT_NAME}-backend --region $AWS_REGION 2>/dev/null || echo "Backend repo exists"
aws ecr create-repository --repository-name ${PROJECT_NAME}-frontend --region $AWS_REGION 2>/dev/null || echo "Frontend repo exists"

# Step 2: Login to ECR
echo "Logging into ECR..."
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Step 3: Build and push images
echo "Building backend image..."
docker build -t ${PROJECT_NAME}-backend:latest -f Dockerfile .

echo "Building frontend image..."
docker build -t ${PROJECT_NAME}-frontend:latest -f Dockerfile.frontend .

echo "Tagging images..."
docker tag ${PROJECT_NAME}-backend:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/${PROJECT_NAME}-backend:latest
docker tag ${PROJECT_NAME}-frontend:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/${PROJECT_NAME}-frontend:latest

echo "Pushing images to ECR..."
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/${PROJECT_NAME}-backend:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/${PROJECT_NAME}-frontend:latest

echo "=== Images pushed successfully ==="
echo ""
echo "Next steps (manual via AWS Console):"
echo "1. Create RDS PostgreSQL instance"
echo "2. Create ElastiCache Redis cluster"  
echo "3. Create ECS cluster and services"
echo "4. Create ALB with target groups"
echo ""
echo "See docs/AWS_DEPLOYMENT.md for detailed instructions"
