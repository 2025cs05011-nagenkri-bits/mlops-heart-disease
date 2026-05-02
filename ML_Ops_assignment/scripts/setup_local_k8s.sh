#!/bin/bash
# ============================================================
# Local Kubernetes deployment for Heart Disease API
# Builds image, applies manifests, waits for rollout, prints
# the public URL and a sample curl.
# ============================================================
set -euo pipefail

IMAGE_TAG="heart-disease-api:v2"
NAMESPACE="heart-disease"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================"
echo "  Heart Disease API - Local K8s Deploy"
echo "  Context: $(kubectl config current-context)"
echo "  Image:   ${IMAGE_TAG}"
echo "============================================"

# ─── Step 1: Ensure trained model exists ────────────────────
echo -e "\n[1/5] Ensuring model artifact exists..."
if [ ! -f "${PROJECT_DIR}/models/heart_model.pkl" ]; then
  echo "  No model found. Running prepare + train..."
  (cd "${PROJECT_DIR}" && python -m src.data.prepare && python -m src.training.train)
else
  echo "  models/heart_model.pkl exists."
fi

# ─── Step 2: Build Docker image ─────────────────────────────
echo -e "\n[2/5] Building Docker image ${IMAGE_TAG}..."
docker build -t "${IMAGE_TAG}" "${PROJECT_DIR}"

# ─── Step 3: Apply K8s manifests ────────────────────────────
echo -e "\n[3/5] Applying Kubernetes manifests..."
kubectl apply -f "${PROJECT_DIR}/k8s/namespace.yaml"
kubectl apply -f "${PROJECT_DIR}/k8s/configmap.yaml"
kubectl apply -f "${PROJECT_DIR}/k8s/deployment.yaml"
kubectl apply -f "${PROJECT_DIR}/k8s/service.yaml"

# ─── Step 4: Wait for rollout ───────────────────────────────
echo -e "\n[4/5] Waiting for deployment to become ready..."
kubectl -n "${NAMESPACE}" rollout status deployment/heart-disease-api --timeout=180s

# ─── Step 5: Show endpoints ─────────────────────────────────
echo -e "\n[5/5] Cluster state:"
kubectl -n "${NAMESPACE}" get all -o wide

echo ""
echo "============================================"
echo "  Deployed!"
echo "============================================"
echo ""
echo "  On Docker Desktop, LoadBalancer is reachable at:"
echo "    http://localhost:8080/health"
echo "    http://localhost:8080/form"
echo ""
echo "  Smoke-test:"
echo "    curl -s http://localhost:8080/health"
echo ""
echo "  Sample prediction:"
cat <<'EOF'
    curl -s -X POST http://localhost:8080/predict \
      -H "Content-Type: application/json" \
      -d '{"age":67,"sex":1,"cp":4,"trestbps":160,"chol":286,"fbs":0,
           "restecg":2,"thalach":108,"exang":1,"oldpeak":1.5,"slope":2,
           "ca":3,"thal":7}'
EOF
echo ""
echo "  Tear down with: ./scripts/cleanup_local_k8s.sh"
echo "============================================"
