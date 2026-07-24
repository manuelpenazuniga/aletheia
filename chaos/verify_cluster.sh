#!/usr/bin/env bash
# Verify the chaos cluster is a real, healthy, three-node CockroachDB cluster.
#
# Run this BEFORE any chaos experiment. E2's claim ("we killed a node and the
# fleet kept remembering") only means something if there were three live nodes
# and a replication factor that tolerates losing one. Asserting that up front is
# cheaper than discovering afterwards that the number was meaningless.
set -euo pipefail

cd "$(dirname "$0")/.."

NODE=${1:-aletheia-chaos-1}
SQL=(docker exec -i "$NODE" ./cockroach sql --insecure --host=localhost:26257)

echo "[verify] querying cluster through $NODE"

echo
echo "--- nodes ---"
"${SQL[@]}" -e "SELECT node_id, address, is_live, ranges FROM crdb_internal.kv_node_status
                JOIN crdb_internal.kv_store_status USING (node_id) ORDER BY node_id;" 2>/dev/null ||
  "${SQL[@]}" -e "SELECT node_id, address, is_live FROM crdb_internal.gossip_nodes ORDER BY node_id;"

LIVE=$("${SQL[@]}" --format=csv -e \
  "SELECT count(*) FROM crdb_internal.gossip_nodes WHERE is_live;" | tail -1 | tr -d '[:space:]')

echo
echo "--- replication ---"
# The chaos claim ("lose a node, keep quorum") only holds at replication factor
# >= 3, so assert it rather than merely printing it.
REPLICAS=$("${SQL[@]}" --format=csv -e \
  "SELECT (crdb_internal.get_zone_config(0)->'num_replicas')::INT8;" 2>/dev/null | tail -1 | tr -d '[:space:]' || true)
if [ -z "${REPLICAS:-}" ] || [ "$REPLICAS" = "NULL" ]; then
  # Fallback for versions where the internal builtin path differs.
  REPLICAS=$("${SQL[@]}" -e "SHOW ZONE CONFIGURATION FOR RANGE default;" \
    | sed -n 's/.*num_replicas = \([0-9]\{1,\}\).*/\1/p' | tail -1)
fi
echo "[verify] num_replicas = ${REPLICAS:-unknown}"

echo
FAIL=0
if [ "$LIVE" != "3" ]; then
  echo "[verify] FAIL — expected 3 live nodes, found: ${LIVE:-unknown}" >&2
  FAIL=1
fi
if [ -z "${REPLICAS:-}" ] || [ "$REPLICAS" -lt 3 ] 2>/dev/null; then
  echo "[verify] FAIL — replication factor ${REPLICAS:-unknown} < 3; losing a node would lose data" >&2
  FAIL=1
fi
if [ "$FAIL" = "0" ]; then
  echo "[verify] OK — 3 live nodes at RF ${REPLICAS}. Losing one leaves quorum (2 of 3)."
else
  exit 1
fi
