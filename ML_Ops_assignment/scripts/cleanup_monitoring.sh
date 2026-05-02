#!/bin/bash
# Tear down the Prometheus + Grafana stack.
set -euo pipefail

NAMESPACE="monitoring"

echo "Uninstalling Helm releases..."
helm uninstall grafana -n "${NAMESPACE}" 2>/dev/null || true
helm uninstall prometheus -n "${NAMESPACE}" 2>/dev/null || true

echo "Deleting namespace ${NAMESPACE}..."
kubectl delete namespace "${NAMESPACE}" --ignore-not-found
echo "Done."
