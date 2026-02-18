# AWS Deployment Guide

This guide deploys the Research Report Generator to AWS using:
- **ECS Fargate** - Container orchestration
- **RDS PostgreSQL** - Database with pgvector
- **ElastiCache Redis** - Caching
- **Application Load Balancer** - Traffic routing

## Prerequisites

1. AWS CLI installed and configured (`aws configure`)
2. Docker installed and running
3. Terraform installed (v1.0+)

## Estimated Monthly Costs

| Service | Spec | Cost |
|---------|------|------|
| ECS Fargate (Backend) | 0.5 vCPU, 1GB | ~$15 |
| ECS Fargate (Frontend) | 0.25 vCPU, 0.5GB | ~$8 |
| RDS PostgreSQL | db.t3.micro | ~$15 |
| ElastiCache Redis | cache.t3.micro | ~$12 |
| ALB | Basic | ~$20 |
| NAT Gateway | Per hour + data | ~$35 |
| **Total** | | **~$105/month** |

## Deployment Steps

### Step 1: Build and Push Docker Images

```bash
# From project root
cd scripts
chmod +x deploy-ecr.sh
./deploy-ecr.sh
```

On Windows (PowerShell):
```powershell
# Get AWS account ID
$AWS_ACCOUNT_ID = aws sts get-caller-identity --query Account --output text
$AWS_REGION = "eu-north-1"

# Login to ECR
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"

# Create repositories
aws ecr create-repository --repository-name research-report-backend --region $AWS_REGION
aws ecr create-repository --repository-name research-report-frontend --region $AWS_REGION

# Build and push backend
docker build -t research-report-backend:latest -f Dockerfile .
docker tag research-report-backend:latest "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/research-report-backend:latest"
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/research-report-backend:latest"

# Build and push frontend
docker build -t research-report-frontend:latest -f Dockerfile.frontend .
docker tag research-report-frontend:latest "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/research-report-frontend:latest"
docker push "$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/research-report-frontend:latest"
```

### Step 2: Configure Terraform Variables

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit `terraform.tfvars` with your values:
```hcl
db_password = "YourSecurePassword123!"
openai_api_key = "sk-..."
tavily_api_key = "tvly-..."
langsmith_api_key = "lsv2_pt_..."
```

### Step 3: Deploy Infrastructure

```bash
cd terraform

# Initialize Terraform
terraform init

# Preview changes
terraform plan

# Deploy (type 'yes' when prompted)
terraform apply
```

This takes about 10-15 minutes.

### Step 4: Verify Deployment

After deployment, Terraform outputs the URLs:

```
alb_dns_name = "research-report-alb-123456789.eu-north-1.elb.amazonaws.com"
api_url = "http://research-report-alb-123456789.eu-north-1.elb.amazonaws.com/health"
frontend_url = "http://research-report-alb-123456789.eu-north-1.elb.amazonaws.com"
```

Test the API:
```bash
curl http://<alb_dns_name>/health
```

Open the frontend in your browser:
```
http://<alb_dns_name>
```

## Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │              Application Load Balancer       │
                    │                   (Public)                   │
                    └─────────────────┬───────────────────────────┘
                                      │
                    ┌─────────────────┴───────────────────────────┐
                    │                                              │
            /research*, /health, /docs                    /* (default)
                    │                                              │
                    ▼                                              ▼
        ┌───────────────────┐                        ┌───────────────────┐
        │   ECS Backend     │                        │   ECS Frontend    │
        │   (Fargate)       │                        │   (Fargate)       │
        │   Port 8000       │                        │   Port 8501       │
        └─────────┬─────────┘                        └───────────────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
┌───────────────┐   ┌───────────────┐
│     RDS       │   │  ElastiCache  │
│  PostgreSQL   │   │    Redis      │
│  (pgvector)   │   │               │
└───────────────┘   └───────────────┘
```

## Troubleshooting

### Check ECS Service Logs
```bash
# Backend logs
aws logs tail /ecs/research-report-backend --follow

# Frontend logs
aws logs tail /ecs/research-report-frontend --follow
```

### Check ECS Task Status
```bash
aws ecs describe-services --cluster research-report-cluster --services research-report-backend research-report-frontend
```

### Common Issues

1. **Tasks not starting**: Check CloudWatch logs for errors
2. **Health checks failing**: Ensure security groups allow traffic
3. **Database connection failed**: Verify RDS security group allows ECS access

## Cleanup

To destroy all resources and stop billing:

```bash
cd terraform
terraform destroy
```

Also delete ECR images:
```bash
aws ecr delete-repository --repository-name research-report-backend --force --region eu-north-1
aws ecr delete-repository --repository-name research-report-frontend --force --region eu-north-1
```

## Enable pgvector Extension

After RDS is created, connect and enable pgvector:

```bash
# Get RDS endpoint from Terraform output
PGHOST=$(terraform output -raw rds_endpoint | cut -d: -f1)

# Connect (you'll need a bastion host or VPN since RDS is private)
psql -h $PGHOST -U postgres -d research

# In psql:
CREATE EXTENSION IF NOT EXISTS vector;
```

For initial setup, you may want to temporarily make RDS publicly accessible, enable the extension, then disable public access.
