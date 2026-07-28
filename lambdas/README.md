# lambdas/ — scheduled memory cycles

The two memory cycles that run serverless, on a schedule, off the fleet's hot path
(the project plan §4, §5.1):

| Function | Handler | Cycle | Default schedule |
|---|---|---|---|
| `aletheia-consolidation` | `lambdas.consolidation_handler.handler` | C2 — knowledge-update: supersede obsolete facts, promote canonical facts | every 10 min |
| `aletheia-gossip` | `lambdas.gossip_handler.handler` | C4 — gossip tick: propagate findings between agents | every 5 min |

Both handlers are **thin**: all logic lives in the portable core
(`core.consolidation.consolidate`, `core.gossip.gossip_tick`); the handler only
bridges the Lambda runtime to it, builds the CockroachDB adapter (and, for gossip,
the Bedrock embedder) lazily from environment variables, and emits structured JSON
logs. They short-circuit when their feature flag is off, and fail loudly if
`ALETHEIA_CRDB_DSN` is unset — no silent no-ops.

## Deploy

One container image (`Dockerfile`) backs both functions; each selects its handler
via `ImageConfig.Command`. EventBridge invokes them on a fixed cadence.

```bash
AWS_REGION=us-east-1 \
EXECUTION_ROLE_ARN=arn:aws:iam::<acct>:role/AletheiaLambdaRole \
ALETHEIA_CRDB_DSN='postgresql://<user>:<pass>@<host>:26257/aletheia?sslmode=verify-full' \
  ./lambdas/deploy_lambdas.sh
```

The script builds for `linux/amd64`, pushes to ECR, creates or updates both
functions, and wires an EventBridge schedule to each. Re-run it to ship a new
image — it updates in place.

### Execution role

The functions assume `EXECUTION_ROLE_ARN`, which needs:

- **`AWSLambdaBasicExecutionRole`** — CloudWatch Logs.
- **`bedrock:InvokeModel`** on the Titan embedding model — the gossip tick embeds
  propagated content.
- **`s3:PutObject`** on the archive bucket — only if you later route metabolic
  forgetting's S3 offload through a Lambda.

CockroachDB Cloud is reached over its public SQL endpoint, so the functions run
**outside a VPC** — no networking config needed. `ALETHEIA_CRDB_DSN` is the
**write** DSN and lives only here and in the ingest service, never in the read
path (the project plan §5.3).

## Verify

```bash
aws lambda invoke --function-name aletheia-consolidation --region <region> /dev/stdout
aws logs tail /aws/lambda/aletheia-consolidation --follow --region <region>
```

A successful consolidation run logs `consolidation.completed` with
`groups`/`supersedes`/`canonical_updates`; a run with the flag off logs
`consolidation.skipped`.
