#!/usr/bin/env bash
# Build → push → deploy the Aletheia demo dashboard to AWS App Runner.
#
# One command to stand up (or update) the public demo URL required by the
# hackathon (the project plan §2, §9.1). Idempotent: creates the App Runner
# service on first run, updates the image in place afterwards. It never runs the
# app itself here — it ships the image and lets App Runner run it.
#
# Prerequisites (provided by the operator when the AWS account is ready):
#   - Docker running; AWS CLI v2 configured with credentials for the target account.
#   - An App Runner ECR access role (a role App Runner assumes to pull from ECR).
#     Create once:  aws iam create-role --role-name AppRunnerECRAccessRole \
#       --assume-role-policy-document file://infra/apprunner-ecr-trust.json
#     then attach the managed policy AWSAppRunnerServicePolicyForECRAccess.
#
# Configuration (environment variables):
#   AWS_REGION            required, e.g. us-east-1
#   ACCESS_ROLE_ARN       required, ARN of the App Runner ECR access role
#   AWS_ACCOUNT_ID        optional, derived via `aws sts get-caller-identity`
#   ECR_REPO              optional, default: aletheia-demo
#   SERVICE_NAME          optional, default: aletheia-demo
#   IMAGE_TAG             optional, default: current git short SHA (else "latest")
#   CPU / MEMORY          optional, default: 1024 (1 vCPU) / 2048 (2 GB)
#   ALETHEIA_DEMO_TOKEN   optional, gates the attack/destructive demo buttons
#
# Usage:
#   AWS_REGION=us-east-1 ACCESS_ROLE_ARN=arn:aws:iam::123:role/AppRunnerECRAccessRole \
#     ALETHEIA_DEMO_TOKEN=some-demo-secret ./infra/deploy_demoapp.sh
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION (e.g. us-east-1)}"
: "${ACCESS_ROLE_ARN:?set ACCESS_ROLE_ARN (App Runner ECR access role ARN)}"

ECR_REPO="${ECR_REPO:-aletheia-demo}"
SERVICE_NAME="${SERVICE_NAME:-aletheia-demo}"
CPU="${CPU:-1024}"
MEMORY="${MEMORY:-2048}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

echo "==> Target: ${IMAGE_URI}  (service: ${SERVICE_NAME}, region: ${AWS_REGION})"

# 1) Ensure the ECR repository exists.
if ! aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "==> Creating ECR repository ${ECR_REPO}"
  aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true >/dev/null
fi

# 2) Authenticate Docker to ECR.
echo "==> Logging Docker in to ECR"
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

# 3) Build for App Runner's architecture (x86_64) and push. --platform matters on
#    Apple Silicon: App Runner will not run an arm64 image.
echo "==> Building and pushing ${IMAGE_URI}"
docker build --platform linux/amd64 -t "${IMAGE_URI}" "${REPO_ROOT}"
docker push "${IMAGE_URI}"

# 4) Assemble the App Runner source/health/instance configuration.
RUNTIME_ENV="{\"DEMOAPP_HOST\":\"0.0.0.0\",\"DEMOAPP_PORT\":\"8080\"}"
if [[ -n "${ALETHEIA_DEMO_TOKEN:-}" ]]; then
  RUNTIME_ENV="{\"DEMOAPP_HOST\":\"0.0.0.0\",\"DEMOAPP_PORT\":\"8080\",\"ALETHEIA_DEMO_TOKEN\":\"${ALETHEIA_DEMO_TOKEN}\"}"
fi

SOURCE_CONFIG=$(cat <<JSON
{
  "ImageRepository": {
    "ImageIdentifier": "${IMAGE_URI}",
    "ImageRepositoryType": "ECR",
    "ImageConfiguration": {
      "Port": "8080",
      "RuntimeEnvironmentVariables": ${RUNTIME_ENV}
    }
  },
  "AutoDeploymentsEnabled": false,
  "AuthenticationConfiguration": { "AccessRoleArn": "${ACCESS_ROLE_ARN}" }
}
JSON
)
HEALTH_CONFIG='{"Protocol":"HTTP","Path":"/healthz","Interval":10,"Timeout":5,"HealthyThreshold":1,"UnhealthyThreshold":5}'
INSTANCE_CONFIG="{\"Cpu\":\"${CPU}\",\"Memory\":\"${MEMORY}\"}"

# 5) Create the service, or update the existing one in place (idempotent).
SERVICE_ARN=$(aws apprunner list-services --region "${AWS_REGION}" \
  --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn | [0]" --output text)

if [[ "${SERVICE_ARN}" == "None" || -z "${SERVICE_ARN}" ]]; then
  echo "==> Creating App Runner service ${SERVICE_NAME}"
  aws apprunner create-service --region "${AWS_REGION}" \
    --service-name "${SERVICE_NAME}" \
    --source-configuration "${SOURCE_CONFIG}" \
    --health-check-configuration "${HEALTH_CONFIG}" \
    --instance-configuration "${INSTANCE_CONFIG}" >/dev/null
  SERVICE_ARN=$(aws apprunner list-services --region "${AWS_REGION}" \
    --query "ServiceSummaryList[?ServiceName=='${SERVICE_NAME}'].ServiceArn | [0]" --output text)
else
  echo "==> Updating existing App Runner service ${SERVICE_NAME}"
  aws apprunner update-service --region "${AWS_REGION}" \
    --service-arn "${SERVICE_ARN}" \
    --source-configuration "${SOURCE_CONFIG}" \
    --health-check-configuration "${HEALTH_CONFIG}" \
    --instance-configuration "${INSTANCE_CONFIG}" >/dev/null
fi

SERVICE_URL=$(aws apprunner describe-service --region "${AWS_REGION}" \
  --service-arn "${SERVICE_ARN}" --query "Service.ServiceUrl" --output text)
echo "==> Deployment triggered. Public URL: https://${SERVICE_URL}"
echo "    (App Runner takes a few minutes to reach RUNNING; watch:"
echo "     aws apprunner describe-service --region ${AWS_REGION} --service-arn ${SERVICE_ARN} --query Service.Status)"
