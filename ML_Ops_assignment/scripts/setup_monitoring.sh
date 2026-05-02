#!/bin/bash
# ============================================================
# Installs Prometheus + Grafana into the local Kubernetes
# cluster and wires Grafana to scrape the heart-disease-api.
# ============================================================
set -euo pipefail

NAMESPACE="monitoring"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

echo "============================================"
echo "  Monitoring stack: Prometheus + Grafana"
echo "  Context: $(kubectl config current-context)"
echo "============================================"

# ─── Step 1: Helm repos ─────────────────────────────────────
echo -e "\n[1/6] Adding Helm repositories..."
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts >/dev/null
helm repo add grafana https://grafana.github.io/helm-charts >/dev/null
helm repo update >/dev/null

# ─── Step 2: Namespace + dashboard ConfigMap ────────────────
echo -e "\n[2/6] Creating namespace + dashboard ConfigMap..."
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
kubectl apply -f "${PROJECT_DIR}/monitoring/dashboard-configmap.yaml"

# ─── Step 3: Install / upgrade Prometheus ───────────────────
echo -e "\n[3/6] Installing Prometheus..."
helm upgrade --install prometheus prometheus-community/prometheus \
  -n "${NAMESPACE}" \
  -f "${PROJECT_DIR}/monitoring/prometheus-values.yaml" \
  --wait --timeout 5m

# ─── Step 4: Install / upgrade Grafana ──────────────────────
echo -e "\n[4/6] Installing Grafana..."
helm upgrade --install grafana grafana/grafana \
  -n "${NAMESPACE}" \
  -f "${PROJECT_DIR}/monitoring/grafana-values.yaml" \
  --wait --timeout 5m

# ─── Step 5: Show stack state ───────────────────────────────
echo -e "\n[5/6] Monitoring namespace state:"
kubectl get all -n "${NAMESPACE}"

# ─── Step 6: Print access instructions ──────────────────────
echo -e "\n[6/6] Access:"
echo ""
echo "  Prometheus UI (port-forward):"
echo "    kubectl -n ${NAMESPACE} port-forward svc/prometheus-server 9090:80"
echo "    open http://localhost:9090"
echo "    -> Status > Targets : look for 'kubernetes-pods' job entries"
echo "       with namespace=heart-disease in state UP."
echo ""
echo "  Grafana UI (port-forward):"
echo "    kubectl -n ${NAMESPACE} port-forward svc/grafana 3000:80"
echo "    open http://localhost:3000"
echo "    -> login: admin / admin"
echo "    -> Dashboards > Heart Disease > Heart Disease API"
echo ""
echo "  Tear down with: ./scripts/cleanup_monitoring.sh"
echo "============================================"
