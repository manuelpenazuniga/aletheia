# Aletheia memory-cycle Lambdas — container image.
#
# ONE image serves BOTH scheduled cycles; each Lambda function selects its handler
# via ImageConfig.Command (set by infra/../lambdas/deploy_lambdas.sh):
#   consolidation (C2) → lambdas.consolidation_handler.handler
#   gossip tick   (C4) → lambdas.gossip_handler.handler
#
# Build from the REPO ROOT so the COPY paths resolve:
#   docker build --platform linux/amd64 -f lambdas/Dockerfile -t <tag> .
FROM public.ecr.aws/lambda/python:3.12

# Runtime dependencies the handlers pull in through the adapters. core/ itself is
# infrastructure-free; psycopg is needed by adapters/cockroach and boto3 by
# adapters/bedrock_embedder (boto3 ships in the base image, but pin it so the
# image is reproducible rather than tracking the base image's floating version).
RUN pip install --no-cache-dir "psycopg[binary]>=3.2" "boto3>=1.34"

# Only what the cycles need at runtime: the portable core, the two adapters they
# build lazily, and the handlers. No web framework, no scenarios, no demo app.
COPY core/     ${LAMBDA_TASK_ROOT}/core/
COPY adapters/ ${LAMBDA_TASK_ROOT}/adapters/
COPY lambdas/  ${LAMBDA_TASK_ROOT}/lambdas/

# Default handler; each function overrides it via ImageConfig.Command.
CMD ["lambdas.consolidation_handler.handler"]
