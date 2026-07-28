#!/usr/bin/env bash
# Build → push → deploy the two Aletheia memory-cycle Lambdas, on a schedule.
#
# One container image (lambdas/Dockerfile) backs both functions; each selects its
# handler via ImageConfig.Command. EventBridge invokes them on a fixed cadence —
# the serverless "consolidation cycle / gossip tick" narrative of the project plan
# (§4, C2/C4). Idempotent: creates the functions + rules on first run, updates the
# image + schedule in place afterwards.
#
# Prerequisites (operator, once the AWS account + CockroachDB Cloud are ready):
#   - Docker running; AWS CLI v2 configured for the target account.
#   - A Lambda EXECUTION ROLE the functions assume. It needs:
#       * AWSLambdaBasicExecutionRole (CloudWatch Logs), and
#       * bedrock:InvokeModel on the Titan embedding model (gossip embeds), and
#       * s3:PutObject on the archive bucket IF you wire metabolic forgetting here.
#     CockroachDB Cloud is reached over its public SQL endpoint, so no VPC config
#     is required (leave the functions outside a VPC).
#   - ALETHEIA_CRDB_DSN — the CockroachDB connection string the cycles write with.
#     This is the WRITE DSN; keep it out of the read path (the project plan §5.3).
#
# Configuration (environment variables):
#   AWS_REGION              required, e.g. us-east-1
#   EXECUTION_ROLE_ARN      required, ARN of the Lambda execution role above
#   ALETHEIA_CRDB_DSN       required, CockroachDB connection string (write path)
#   AWS_ACCOUNT_ID          optional, derived via `aws sts get-caller-identity`
#   ECR_REPO                optional, default: aletheia-lambdas
#   IMAGE_TAG               optional, default: current git short SHA (else "latest")
#   CONSOLIDATION_RATE      optional, default: "rate(10 minutes)"
#   GOSSIP_RATE             optional, default: "rate(5 minutes)"
#   MEMORY_MB / TIMEOUT_S   optional, default: 512 / 120
#   ALETHEIA_EMBEDDING_MODEL_ID / ALETHEIA_EMBEDDING_DIM  optional, passed through
#
# Usage:
#   AWS_REGION=us-east-1 EXECUTION_ROLE_ARN=arn:aws:iam::123:role/AletheiaLambdaRole \
#     ALETHEIA_CRDB_DSN='postgresql://…' ./lambdas/deploy_lambdas.sh
set -euo pipefail

: "${AWS_REGION:?set AWS_REGION (e.g. us-east-1)}"
: "${EXECUTION_ROLE_ARN:?set EXECUTION_ROLE_ARN (Lambda execution role)}"
: "${ALETHEIA_CRDB_DSN:?set ALETHEIA_CRDB_DSN (CockroachDB write connection string)}"

ECR_REPO="${ECR_REPO:-aletheia-lambdas}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD 2>/dev/null || echo latest)}"
MEMORY_MB="${MEMORY_MB:-512}"
TIMEOUT_S="${TIMEOUT_S:-120}"
CONSOLIDATION_RATE="${CONSOLIDATION_RATE:-rate(10 minutes)}"
GOSSIP_RATE="${GOSSIP_RATE:-rate(5 minutes)}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

# Environment handed to both functions. Only ALETHEIA_CRDB_DSN is required; the
# rest fall back to AletheiaConfig defaults unless the operator overrides them.
ENV_VARS="ALETHEIA_CRDB_DSN=${ALETHEIA_CRDB_DSN}"
[[ -n "${ALETHEIA_EMBEDDING_MODEL_ID:-}" ]] && ENV_VARS="${ENV_VARS},ALETHEIA_EMBEDDING_MODEL_ID=${ALETHEIA_EMBEDDING_MODEL_ID}"
[[ -n "${ALETHEIA_EMBEDDING_DIM:-}" ]] && ENV_VARS="${ENV_VARS},ALETHEIA_EMBEDDING_DIM=${ALETHEIA_EMBEDDING_DIM}"

echo "==> Target image: ${IMAGE_URI}  (region: ${AWS_REGION})"

# 1) Ensure the ECR repo, authenticate, build for Lambda's arch (x86_64), push.
if ! aws ecr describe-repositories --repository-names "${ECR_REPO}" --region "${AWS_REGION}" >/dev/null 2>&1; then
  echo "==> Creating ECR repository ${ECR_REPO}"
  aws ecr create-repository --repository-name "${ECR_REPO}" --region "${AWS_REGION}" \
    --image-scanning-configuration scanOnPush=true >/dev/null
fi
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"
echo "==> Building and pushing ${IMAGE_URI}"
docker build --platform linux/amd64 -f "${REPO_ROOT}/lambdas/Dockerfile" -t "${IMAGE_URI}" "${REPO_ROOT}"
docker push "${IMAGE_URI}"

# 2) Create-or-update each function, then schedule it via an EventBridge rule.
#    fn <name> <handler-command> <schedule-expression>
deploy_fn() {
  local name="$1" handler="$2" schedule="$3"
  local fn_arn

  if aws lambda get-function --function-name "${name}" --region "${AWS_REGION}" >/dev/null 2>&1; then
    echo "==> Updating function ${name}"
    aws lambda update-function-code --function-name "${name}" --region "${AWS_REGION}" \
      --image-uri "${IMAGE_URI}" >/dev/null
    aws lambda wait function-updated --function-name "${name}" --region "${AWS_REGION}"
    aws lambda update-function-configuration --function-name "${name}" --region "${AWS_REGION}" \
      --image-config "Command=${handler}" \
      --memory-size "${MEMORY_MB}" --timeout "${TIMEOUT_S}" \
      --environment "Variables={${ENV_VARS}}" >/dev/null
  else
    echo "==> Creating function ${name}"
    aws lambda create-function --function-name "${name}" --region "${AWS_REGION}" \
      --package-type Image --code "ImageUri=${IMAGE_URI}" \
      --image-config "Command=${handler}" \
      --role "${EXECUTION_ROLE_ARN}" \
      --memory-size "${MEMORY_MB}" --timeout "${TIMEOUT_S}" \
      --environment "Variables={${ENV_VARS}}" >/dev/null
  fi
  aws lambda wait function-updated --function-name "${name}" --region "${AWS_REGION}"
  fn_arn=$(aws lambda get-function --function-name "${name}" --region "${AWS_REGION}" \
    --query "Configuration.FunctionArn" --output text)

  echo "==> Scheduling ${name}: ${schedule}"
  aws events put-rule --name "${name}-schedule" --region "${AWS_REGION}" \
    --schedule-expression "${schedule}" --state ENABLED >/dev/null
  # Idempotent permission (ignore 'already exists').
  aws lambda add-permission --function-name "${name}" --region "${AWS_REGION}" \
    --statement-id "${name}-eventbridge" --action lambda:InvokeFunction \
    --principal events.amazonaws.com \
    --source-arn "arn:aws:events:${AWS_REGION}:${AWS_ACCOUNT_ID}:rule/${name}-schedule" \
    >/dev/null 2>&1 || true
  aws events put-targets --rule "${name}-schedule" --region "${AWS_REGION}" \
    --targets "Id=1,Arn=${fn_arn}" >/dev/null
}

deploy_fn "aletheia-consolidation" "lambdas.consolidation_handler.handler" "${CONSOLIDATION_RATE}"
deploy_fn "aletheia-gossip"        "lambdas.gossip_handler.handler"        "${GOSSIP_RATE}"

echo "==> Done. Both cycles are deployed and scheduled."
echo "    Tail logs:  aws logs tail /aws/lambda/aletheia-consolidation --follow --region ${AWS_REGION}"
echo "    Invoke now: aws lambda invoke --function-name aletheia-consolidation --region ${AWS_REGION} /dev/stdout"
