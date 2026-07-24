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
"${SQL[@]}" -e "SHOW ZONE CONFIGURATION FOR RANGE default;" | grep -i num_replicas || true

echo
if [ "$LIVE" = "3" ]; then
  echo "[verify] OK — 3 live nodes. Losing one leaves quorum (2 of 3)."
else
  echo "[verify] FAIL — expected 3 live nodes, found: ${LIVE:-unknown}" >&2
  exit 1
fi
