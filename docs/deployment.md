# Deployment Guide

This document provides detailed instructions for deploying the Atrade system in different environments.

## Local Development Setup

### macOS

1. Install Prerequisites:
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.8+
brew install python

# Install PostgreSQL
brew install postgresql@12
brew services start postgresql@12

# Install Redis
brew install redis
brew services start redis
```

2. Create and Activate Virtual Environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

3. Install Dependencies:
```bash
# Install base dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

4. Configure Environment:
```bash
# Copy example environment file
cp .env.example .env

# Edit .env file with your settings
nano .env
```

5. Initialize Database:
```bash
# Create database
createdb trading_system

# Run migrations
python -m atrade.db.migrations upgrade
```

6. Run the Application:
```bash
# Start the trading system
atrade start

# Or run in development mode
python -m atrade.cli start --debug
```

### Windows

1. Install Prerequisites:
```powershell
# Install Python 3.8+ from python.org
# Install PostgreSQL from postgresql.org
# Install Redis from github.com/microsoftarchive/redis/releases
```

2. Create and Activate Virtual Environment:
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

3. Install Dependencies:
```powershell
# Install base dependencies
pip install -r requirements.txt

# Install development dependencies
pip install -e ".[dev]"
```

4. Configure Environment:
```powershell
# Copy example environment file
copy .env.example .env

# Edit .env file with your settings
notepad .env
```

5. Initialize Database:
```powershell
# Create database using pgAdmin or psql
createdb trading_system

# Run migrations
python -m atrade.db.migrations upgrade
```

6. Run the Application:
```powershell
# Start the trading system
atrade start

# Or run in development mode
python -m atrade.cli start --debug
```

## Docker Deployment

### Using Docker Compose (Recommended)

1. Build and Start Services:
```bash
# Build images
docker-compose build

# Start services
docker-compose up -d
```

2. Check Service Status:
```bash
# View logs
docker-compose logs -f

# Check service status
docker-compose ps
```

3. Stop Services:
```bash
docker-compose down
```

### Manual Docker Deployment

1. Build Images:
```bash
# Build application image
docker build -t atrade-app .

# Build database image
docker build -t atrade-db -f Dockerfile.db .
```

2. Create Network:
```bash
docker network create atrade-network
```

3. Run Services:
```bash
# Run database
docker run -d \
  --name atrade-db \
  --network atrade-network \
  -e POSTGRES_DB=trading_system \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  atrade-db

# Run Redis
docker run -d \
  --name atrade-redis \
  --network atrade-network \
  -p 6379:6379 \
  redis:6

# Run application
docker run -d \
  --name atrade-app \
  --network atrade-network \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  atrade-app
```

## Production Deployment

### Using Docker Swarm

1. Initialize Swarm:
```bash
docker swarm init
```

2. Deploy Stack:
```bash
docker stack deploy -c docker-compose.prod.yml atrade
```

3. Monitor Services:
```bash
docker service ls
docker service logs atrade_app
```

### Using Kubernetes

1. Create Namespace:
```bash
kubectl create namespace atrade
```

2. Apply Configurations:
```bash
# Apply secrets
kubectl apply -f k8s/secrets.yaml

# Apply configmaps
kubectl apply -f k8s/configmaps.yaml

# Apply deployments
kubectl apply -f k8s/deployments.yaml

# Apply services
kubectl apply -f k8s/services.yaml
```

3. Monitor Deployment:
```bash
kubectl get pods -n atrade
kubectl get services -n atrade
```

## Monitoring and Maintenance

### Health Checks

1. API Health Check:
```bash
curl http://localhost:8000/api/health
```

2. Database Health Check:
```bash
psql -h localhost -U postgres -d trading_system -c "SELECT 1;"
```

3. Redis Health Check:
```bash
redis-cli ping
```

### Backup and Recovery

1. Database Backup:
```bash
# Create backup
pg_dump -h localhost -U postgres trading_system > backup.sql

# Restore from backup
psql -h localhost -U postgres trading_system < backup.sql
```

2. Configuration Backup:
```bash
# Backup configuration
cp .env .env.backup
cp atrade/config/config.yaml atrade/config/config.yaml.backup
```

### Logging

1. View Application Logs:
```bash
# Docker
docker-compose logs -f app

# Kubernetes
kubectl logs -f deployment/atrade-app -n atrade
```

2. View Database Logs:
```bash
# Docker
docker-compose logs -f db

# Kubernetes
kubectl logs -f deployment/atrade-db -n atrade
```

## Troubleshooting

### Common Issues

1. Database Connection Issues:
- Check database service status
- Verify credentials in .env file
- Check network connectivity

2. Redis Connection Issues:
- Check Redis service status
- Verify Redis configuration
- Check network connectivity

3. Application Issues:
- Check application logs
- Verify environment variables
- Check resource usage

### Performance Optimization

1. Database Optimization:
```sql
-- Create indexes
CREATE INDEX idx_trades_date ON trades(date);
CREATE INDEX idx_positions_symbol ON positions(symbol);
```

2. Redis Optimization:
```bash
# Configure Redis memory limits
redis-cli config set maxmemory 2gb
redis-cli config set maxmemory-policy allkeys-lru
```

3. Application Optimization:
- Adjust worker processes
- Configure connection pools
- Enable caching 