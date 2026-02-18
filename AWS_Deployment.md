# AWS Deployment Guide

This guide documents the AWS deployment of the Intelligent Research Report Generator.

## Live Deployment

| Component | URL |
|-----------|-----|
| **Frontend** | http://research-report-alb-2064123991.eu-north-1.elb.amazonaws.com |
| **API Docs** | http://research-report-alb-2064123991.eu-north-1.elb.amazonaws.com/docs |
| **Health Check** | http://research-report-alb-2064123991.eu-north-1.elb.amazonaws.com/health |

## Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │              Application Load Balancer              │
                    │        research-report-alb-2064123991               │
                    │                                                     │
Internet ──────────▶│   ┌──────────────────┐    ┌──────────────────┐    │
                    │   │ /research*       │    │ /* (default)     │    │
                    │   │ /health          │    │ → Frontend       │    │
                    │   │ /docs            │    │   (Streamlit)    │    │
                    │   │ → Backend        │    │   Port 8501      │    │
                    │   │   (FastAPI)      │    │                  │    │
                    │   │   Port 8000      │    │                  │    │
                    │   └──────────────────┘    └──────────────────┘    │
                    └─────────────────────────────────────────────────────┘
                                        │
                    ┌───────────────────┴───────────────────┐
                    ▼                                       ▼
            ┌─────────────────┐                   ┌─────────────────┐
            │  ECS Fargate    │                   │  ECS Fargate    │
            │  Backend        │                   │  Frontend       │
            │  research-report│                   │  research-report│
            │  -backend       │                   │  -frontend      │
            │                 │                   │                 │
            │  0.5 vCPU       │                   │  0.25 vCPU      │
            │  1024 MB        │                   │  512 MB         │
            └────────┬────────┘                   └─────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐     ┌─────────────────┐
│      RDS        │     │   ElastiCache   │
│   PostgreSQL    │     │     Redis       │
│   research-     │     │   research-     │
│   report-db     │     │   report-redis  │
│                 │     │                 │
│   v16.6         │     │   Redis 7       │
│   db.t3.micro   │     │  cache.t3.micro │
│   + pgvector    │     │                 │
└─────────────────┘     └─────────────────┘
```

## Infrastructure Components

### VPC Configuration
- **CIDR:** 10.0.0.0/16
- **Public Subnets:** 10.0.1.0/24, 10.0.2.0/24 (2 AZs)
- **Private Subnets:** 10.0.10.0/24, 10.0.11.0/24 (2 AZs)
- **NAT Gateway:** For outbound internet access from private subnets
- **Internet Gateway:** For public subnet internet access

### Security Groups
| Name | Inbound Rules |
|------|---------------|
| research-report-alb-sg | 80, 443 from 0.0.0.0/0 |
| research-report-ecs-sg | 8000, 8501 from ALB SG |
| research-report-rds-sg | 5432 from ECS SG |
| research-report-redis-sg | 6379 from ECS SG |

### ECS Configuration
- **Cluster:** research-report-cluster
- **Launch Type:** Fargate
- **Platform Version:** 1.4.0

### RDS Configuration
- **Instance:** research-report-db
- **Engine:** PostgreSQL 16.6
- **Instance Class:** db.t3.micro
- **Storage:** 20 GB gp2
- **Extensions:** pgvector (for embeddings)

### ElastiCache Configuration
- **Cluster:** research-report-redis
- **Engine:** Redis 7
- **Node Type:** cache.t3.micro
- **Nodes:** 1

## Prerequisites

1. **AWS CLI** installed and configured
   ```powershell
   winget install Amazon.AWSCLI
   aws configure
   ```

2. **Terraform** installed (v1.0+)
   ```powershell
   winget install Hashicorp.Terraform
   ```

3. **Docker** installed and running

4. **IAM Permissions** - Your user needs:
   - AmazonEC2ContainerRegistryFullAccess
   - AmazonECS_FullAccess
   - AmazonRDSFullAccess
   - AmazonElastiCacheFullAccess
   - AmazonVPCFullAccess
   - IAMFullAccess
   - ElasticLoadBalancingFullAccess
   - CloudWatchLogsFullAccess

## Deployment Steps

### Step 1: Create ECR Repositories

```powershell
$AWS_REGION = "eu-north-1"
$AWS_ACCOUNT_ID = aws sts get-caller-identity --query Account --output text

# Create repositories
aws ecr create-repository --repository-name research-report-backend --region $AWS_REGION
aws ecr create-repository --repository-name research-report-frontend --region $AWS_REGION
```

### Step 2: Build and Push Docker Images

```powershell
# Login to ECR
$password = aws ecr get-login-password --region $AWS_REGION
docker login --username AWS --password $password "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# Build and push backend
docker build -t research-report-backend:latest -f Dockerfile .
docker tag research-report-backend:latest "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/research-report-backend:latest"
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/research-report-backend:latest"

# Build and push frontend
docker build -t research-report-frontend:latest -f Dockerfile.frontend .
docker tag research-report-frontend:latest "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/research-report-frontend:latest"
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/research-report-frontend:latest"
```

### Step 3: Configure Terraform

```powershell
cd terraform
copy terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars`:
```hcl
aws_region = "eu-north-1"
project_name = "research-report"
environment = "prod"

db_password = "YourSecurePassword123!"
openai_api_key = "sk-..."
tavily_api_key = "tvly-..."
langsmith_api_key = "lsv2_pt_..."
huggingface_api_key = "hf_..."
```

### Step 4: Deploy Infrastructure

```powershell
cd terraform
terraform init
terraform plan    # Preview changes
terraform apply   # Deploy (type 'yes')
```

Deployment takes ~15 minutes. Outputs include:
- `alb_dns_name` - Your application URL
- `rds_endpoint` - Database endpoint
- `redis_endpoint` - Cache endpoint

### Step 5: Verify Deployment

```powershell
# Check health
Invoke-WebRequest -Uri "http://research-report-alb-2064123991.eu-north-1.elb.amazonaws.com/health" -UseBasicParsing

# Check ECS services
aws ecs describe-services --cluster research-report-cluster --services research-report-backend research-report-frontend --region eu-north-1 --query "services[].{name:serviceName,running:runningCount,desired:desiredCount}"
```

## Operations

### View Logs

```powershell
# Backend logs
aws logs tail /ecs/research-report-backend --region eu-north-1 --follow

# Frontend logs
aws logs tail /ecs/research-report-frontend --region eu-north-1 --follow

# Last 10 minutes
aws logs tail /ecs/research-report-backend --region eu-north-1 --since 10m
```

### Redeploy After Code Changes

```powershell
# Rebuild and push
docker build -t research-report-backend:latest -f Dockerfile .
docker tag research-report-backend:latest 383579119887.dkr.ecr.eu-north-1.amazonaws.com/research-report-backend:latest
docker push 383579119887.dkr.ecr.eu-north-1.amazonaws.com/research-report-backend:latest

# Force ECS to pull new image
aws ecs update-service --cluster research-report-cluster --service research-report-backend --force-new-deployment --region eu-north-1
```

### Scale Services

```powershell
# Scale to 0 (stop containers, keep infrastructure)
aws ecs update-service --cluster research-report-cluster --service research-report-backend --desired-count 0 --region eu-north-1
aws ecs update-service --cluster research-report-cluster --service research-report-frontend --desired-count 0 --region eu-north-1

# Scale back to 1
aws ecs update-service --cluster research-report-cluster --service research-report-backend --desired-count 1 --region eu-north-1
aws ecs update-service --cluster research-report-cluster --service research-report-frontend --desired-count 1 --region eu-north-1

# Scale for high availability (2 instances)
aws ecs update-service --cluster research-report-cluster --service research-report-backend --desired-count 2 --region eu-north-1
```

## Cost Management

### Estimated Monthly Costs

| Service | Spec | Cost |
|---------|------|------|
| NAT Gateway | Per hour + data | ~$35 |
| ALB | Basic usage | ~$20 |
| RDS PostgreSQL | db.t3.micro | ~$15 |
| ECS Fargate (backend) | 0.5 vCPU, 1GB | ~$15 |
| ECS Fargate (frontend) | 0.25 vCPU, 0.5GB | ~$8 |
| ElastiCache Redis | cache.t3.micro | ~$12 |
| **Total** | | **~$105/month** |

### Stop All Billing

```powershell
cd terraform
terraform destroy
```

This removes ALL resources. Redeploy with `terraform apply`.

### Partial Cost Reduction

Scale ECS to 0 to save ~$23/month while keeping infrastructure:
```powershell
aws ecs update-service --cluster research-report-cluster --service research-report-backend --desired-count 0 --region eu-north-1
aws ecs update-service --cluster research-report-cluster --service research-report-frontend --desired-count 0 --region eu-north-1
```

**Note:** RDS, ElastiCache, NAT Gateway, and ALB still incur charges (~$82/month).

## Troubleshooting

### Container Not Starting

1. Check logs:
   ```powershell
   aws logs tail /ecs/research-report-backend --region eu-north-1 --since 5m
   ```

2. Check task status:
   ```powershell
   aws ecs list-tasks --cluster research-report-cluster --service-name research-report-backend --desired-status STOPPED --region eu-north-1
   ```

3. Common issues:
   - Missing environment variables → Check Terraform task definition
   - Image not found → Verify ECR push succeeded
   - Health check failing → Verify `/health` endpoint works

### ECR Push Fails (403 Forbidden)

ECR login expires after 12 hours. Re-authenticate:
```powershell
$password = aws ecr get-login-password --region eu-north-1
docker login --username AWS --password $password 383579119887.dkr.ecr.eu-north-1.amazonaws.com
```

### Database Connection Issues

1. Verify security group allows ECS → RDS on port 5432
2. Check DATABASE_URL format in task definition
3. Ensure RDS is in private subnet with ECS access

### Terraform Errors

- **"No configuration files"** → Run from `terraform/` directory
- **"Resource already exists"** → Import existing resource or destroy first
- **"Access denied"** → Check IAM permissions

## Files Reference

```
terraform/
├── main.tf              # All infrastructure definitions
├── terraform.tfvars     # Your secrets (not committed)
└── terraform.tfvars.example

Dockerfile               # Backend container
Dockerfile.frontend      # Frontend container
.dockerignore           # Files excluded from Docker build
```

## Environment Variables

### Backend (ECS Task Definition)

| Variable | Source |
|----------|--------|
| DATABASE_URL | Constructed from RDS endpoint |
| REDIS_URL | Constructed from ElastiCache endpoint |
| OPENAI_API_KEY | terraform.tfvars |
| TAVILY_API_KEY | terraform.tfvars |
| LANGSMITH_API_KEY | terraform.tfvars |
| HUGGINGFACE_API_KEY | terraform.tfvars |
| LANGCHAIN_TRACING_V2 | "true" |
| LANGCHAIN_PROJECT | "research-report-generator" |

### Frontend (ECS Task Definition)

| Variable | Source |
|----------|--------|
| API_URL | ALB DNS name |
