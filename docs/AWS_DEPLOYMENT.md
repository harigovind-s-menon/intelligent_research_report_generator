# AWS Deployment Guide

This guide walks through deploying the Research Report Generator to AWS.

## Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │                  Application Load Balancer          │
                    │     ┌─────────────────┐    ┌─────────────────┐     │
Internet ──────────▶│     │ :80/api/*       │    │ :80/*           │     │
                    │     │ → Backend:8000  │    │ → Frontend:8501 │     │
                    │     └─────────────────┘    └─────────────────┘     │
                    └─────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    ▼                   ▼                   ▼
            ┌─────────────┐    ┌─────────────────┐   ┌─────────────┐
            │ ECS Fargate │    │  ECS Fargate    │   │             │
            │  Backend    │    │   Frontend      │   │             │
            │  (FastAPI)  │    │  (Streamlit)    │   │             │
            └─────────────┘    └─────────────────┘   │             │
                    │                                 │             │
                    ▼                                 │             │
            ┌─────────────┐                          │             │
            │    RDS      │◀─────────────────────────┘             │
            │ PostgreSQL  │                                        │
            │ (pgvector)  │                                        │
            └─────────────┘                                        │
                    │                                              │
                    ▼                                              │
            ┌─────────────┐                                        │
            │ ElastiCache │◀───────────────────────────────────────┘
            │   Redis     │
            └─────────────┘
```

## Prerequisites

1. AWS CLI installed and configured
2. Docker installed locally
3. Your AWS account with appropriate permissions

## Step 1: Set Up ECR Repositories

```bash
# Set your region
export AWS_REGION=eu-north-1
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Create ECR repositories
aws ecr create-repository --repository-name research-report-backend --region $AWS_REGION
aws ecr create-repository --repository-name research-report-frontend --region $AWS_REGION

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

## Step 2: Build and Push Docker Images

```bash
# Build backend
docker build -t research-report-backend:latest -f Dockerfile .

# Build frontend
docker build -t research-report-frontend:latest -f Dockerfile.frontend .

# Tag images
docker tag research-report-backend:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/research-report-backend:latest
docker tag research-report-frontend:latest $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/research-report-frontend:latest

# Push images
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/research-report-backend:latest
docker push $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/research-report-frontend:latest
```

## Step 3: Create VPC and Security Groups (via Console)

1. Go to **VPC Console** → Create VPC
   - Name: `research-report-vpc`
   - IPv4 CIDR: `10.0.0.0/16`
   - Create with public and private subnets in 2 AZs

2. Create Security Groups:

   **ALB Security Group** (`research-report-alb-sg`):
   - Inbound: HTTP (80) from 0.0.0.0/0
   - Inbound: HTTPS (443) from 0.0.0.0/0

   **ECS Security Group** (`research-report-ecs-sg`):
   - Inbound: 8000 from ALB SG
   - Inbound: 8501 from ALB SG

   **RDS Security Group** (`research-report-rds-sg`):
   - Inbound: 5432 from ECS SG

   **Redis Security Group** (`research-report-redis-sg`):
   - Inbound: 6379 from ECS SG

## Step 4: Create RDS PostgreSQL

1. Go to **RDS Console** → Create database
   - Engine: PostgreSQL 16
   - Template: Free tier (or production for real use)
   - DB instance identifier: `research-report-db`
   - Master username: `postgres`
   - Master password: (save this!)
   - Instance: `db.t3.micro` (free tier)
   - VPC: `research-report-vpc`
   - Security group: `research-report-rds-sg`
   - Initial database: `research_reports`

2. After creation, connect and enable pgvector:
```bash
psql -h <rds-endpoint> -U postgres -d research_reports
CREATE EXTENSION vector;
```

## Step 5: Create ElastiCache Redis

1. Go to **ElastiCache Console** → Create cluster
   - Cluster engine: Redis
   - Name: `research-report-cache`
   - Node type: `cache.t3.micro`
   - Number of replicas: 0 (for dev)
   - VPC: `research-report-vpc`
   - Security group: `research-report-redis-sg`

## Step 6: Create ECS Cluster

1. Go to **ECS Console** → Create cluster
   - Name: `research-report-cluster`
   - Infrastructure: AWS Fargate

## Step 7: Store Secrets in AWS Secrets Manager

```bash
# Create secret with all environment variables
aws secretsmanager create-secret \
    --name research-report/env \
    --region $AWS_REGION \
    --secret-string '{
        "OPENAI_API_KEY": "sk-...",
        "TAVILY_API_KEY": "tvly-...",
        "HUGGINGFACE_API_KEY": "hf_...",
        "LANGSMITH_API_KEY": "lsv2_...",
        "DATABASE_URL": "postgresql://postgres:PASSWORD@RDS_ENDPOINT:5432/research_reports",
        "REDIS_URL": "redis://ELASTICACHE_ENDPOINT:6379"
    }'
```

## Step 8: Create Task Definitions

### Backend Task Definition (`backend-task.json`):
```json
{
    "family": "research-report-backend",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "512",
    "memory": "1024",
    "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
    "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskRole",
    "containerDefinitions": [
        {
            "name": "backend",
            "image": "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/research-report-backend:latest",
            "portMappings": [
                {
                    "containerPort": 8000,
                    "protocol": "tcp"
                }
            ],
            "secrets": [
                {"name": "OPENAI_API_KEY", "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:research-report/env:OPENAI_API_KEY::"},
                {"name": "TAVILY_API_KEY", "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:research-report/env:TAVILY_API_KEY::"},
                {"name": "HUGGINGFACE_API_KEY", "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:research-report/env:HUGGINGFACE_API_KEY::"},
                {"name": "LANGSMITH_API_KEY", "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:research-report/env:LANGSMITH_API_KEY::"},
                {"name": "DATABASE_URL", "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:research-report/env:DATABASE_URL::"},
                {"name": "REDIS_URL", "valueFrom": "arn:aws:secretsmanager:REGION:ACCOUNT_ID:secret:research-report/env:REDIS_URL::"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/research-report-backend",
                    "awslogs-region": "REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
```

### Frontend Task Definition (`frontend-task.json`):
```json
{
    "family": "research-report-frontend",
    "networkMode": "awsvpc",
    "requiresCompatibilities": ["FARGATE"],
    "cpu": "256",
    "memory": "512",
    "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
    "containerDefinitions": [
        {
            "name": "frontend",
            "image": "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/research-report-frontend:latest",
            "portMappings": [
                {
                    "containerPort": 8501,
                    "protocol": "tcp"
                }
            ],
            "environment": [
                {"name": "API_URL", "value": "http://INTERNAL_ALB_DNS/api"}
            ],
            "logConfiguration": {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": "/ecs/research-report-frontend",
                    "awslogs-region": "REGION",
                    "awslogs-stream-prefix": "ecs"
                }
            }
        }
    ]
}
```

## Step 9: Create Application Load Balancer

1. Go to **EC2 Console** → Load Balancers → Create
   - Type: Application Load Balancer
   - Name: `research-report-alb`
   - Scheme: Internet-facing
   - VPC: `research-report-vpc`
   - Subnets: Select public subnets
   - Security group: `research-report-alb-sg`

2. Create Target Groups:
   - `research-report-backend-tg` (port 8000, health check: `/health`)
   - `research-report-frontend-tg` (port 8501, health check: `/_stcore/health`)

3. Configure Listener Rules:
   - Path `/api/*` → `research-report-backend-tg`
   - Default → `research-report-frontend-tg`

## Step 10: Create ECS Services

```bash
# Create backend service
aws ecs create-service \
    --cluster research-report-cluster \
    --service-name backend \
    --task-definition research-report-backend \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=backend,containerPort=8000"

# Create frontend service
aws ecs create-service \
    --cluster research-report-cluster \
    --service-name frontend \
    --task-definition research-report-frontend \
    --desired-count 1 \
    --launch-type FARGATE \
    --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
    --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=frontend,containerPort=8501"
```

## Step 11: Update Frontend to Use ALB

The frontend needs to know the backend URL. Update the Streamlit app's default API URL to point to the ALB:

```python
API_URL = st.sidebar.text_input("API URL", value="http://ALB_DNS_NAME/api")
```

Or set it via environment variable in the task definition.

## Step 12: Verify Deployment

1. Get ALB DNS name from EC2 Console
2. Visit `http://ALB_DNS_NAME` — should show Streamlit frontend
3. Check health: `http://ALB_DNS_NAME/api/health`
4. Run a research query through the UI

## Estimated Costs (Monthly)

| Service | Spec | Est. Cost |
|---------|------|-----------|
| ECS Fargate (backend) | 0.5 vCPU, 1GB | ~$15 |
| ECS Fargate (frontend) | 0.25 vCPU, 0.5GB | ~$8 |
| RDS PostgreSQL | db.t3.micro | ~$15 |
| ElastiCache Redis | cache.t3.micro | ~$12 |
| ALB | Basic usage | ~$20 |
| **Total** | | **~$70/month** |

## Quick Start (Minimal Setup)

For a faster, cheaper deployment, you can skip ElastiCache and use a single container:

1. Use SQLite instead of PostgreSQL (not recommended for production)
2. Skip Redis caching (queries won't be cached)
3. Run both frontend and backend in one container

This reduces cost to ~$25/month but sacrifices scalability and reliability.
