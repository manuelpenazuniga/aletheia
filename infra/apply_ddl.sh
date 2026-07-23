#!/usr/bin/env bash
# Bring up (if needed) the local CockroachDB node and apply infra/ddl.sql.
# Against the cloud cluster, set ALETHEIA_CRDB_DSN to the cloud DSN first —
# this script only waits for the local container when the DSN is localhost.
set -euo pipefail

cd "$(dirname "$0")/.."

# shellcheck disable=SC1091
[ -f .env ] && set -a && . ./.env && set +a

DSN="${ALETHEIA_CRDB_DSN:-postgresql://root@localhost:26257/aletheia?sslmode=disable}"

if [[ "$DSN" == *localhost* || "$DSN" == *127.0.0.1* ]]; then
  echo "[apply_ddl] local target — ensuring container is up"
  docker compose up -d
  echo -n "[apply_ddl] waiting for CockroachDB"
  for _ in $(seq 1 60); do
    if docker compose exec -T crdb ./cockroach sql --insecure -e "SELECT 1" >/dev/null 2>&1; then
      echo " ready"
      break
    fi
    echo -n "."
    sleep 2
  done
fi

exec python infra/apply_ddl.py --dsn "$DSN"
