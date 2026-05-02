#!/bin/bash
# ============================================================
# Tear down the local Heart Disease deployment.
# Deletes the namespace (and everything inside it).
# ============================================================
set -euo pipefail

NAMESPACE="heart-disease"

echo "Deleting namespace ${NAMESPACE} (this removes all resources inside)..."
kubectl delete namespace "${NAMESPACE}" --ignore-not-found

echo "Done."
