#!/bin/bash
# recover_docker.sh
#
# Single utility for the "uninstall Docker -> reinstall Docker -> bring the
# whole MLOps stack back" workflow on macOS.
#
# Usage:
#   ./scripts/recover_docker.sh                # interactive menu
#   ./scripts/recover_docker.sh uninstall      # uninstall Docker Desktop
#   ./scripts/recover_docker.sh install-check  # open download page + wait
#   ./scripts/recover_docker.sh recover        # rebuild image + redeploy
#   ./scripts/recover_docker.sh all            # uninstall + install + recover
#
# Safe to re-run; each step is idempotent and confirms destructive actions.

set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

C_RST=$'\033[0m'; C_BLU=$'\033[1;34m'; C_GRN=$'\033[1;32m'
C_YEL=$'\033[1;33m'; C_RED=$'\033[1;31m'

log()  { echo "${C_BLU}[*]${C_RST} $*"; }
ok()   { echo "${C_GRN}[+]${C_RST} $*"; }
warn() { echo "${C_YEL}[!]${C_RST} $*"; }
err()  { echo "${C_RED}[x]${C_RST} $*" >&2; }

confirm() {
  local prompt="${1:-Continue?}" reply
  read -r -p "$prompt [y/N] " reply
  [[ "$reply" =~ ^[Yy]$ ]]
}

wait_for_docker() {
  log "Waiting for Docker daemon to come up (timeout 180s)..."
  for i in $(seq 1 60); do
    if docker info >/dev/null 2>&1; then
      ok "Docker daemon is responding."
      return 0
    fi
    sleep 3
  done
  err "Docker daemon did not start within 3 minutes."
  return 1
}

wait_for_k8s() {
  log "Waiting for Kubernetes context to be ready (timeout 240s)..."
  for i in $(seq 1 80); do
    if kubectl get nodes >/dev/null 2>&1 && \
       kubectl get nodes 2>/dev/null | grep -q " Ready "; then
      ok "Kubernetes node is Ready."
      return 0
    fi
    sleep 3
  done
  err "Kubernetes did not become Ready within 4 minutes."
  err "Open Docker Desktop -> Settings -> Kubernetes -> Enable Kubernetes."
  return 1
}

cmd_uninstall() {
  log "Step 1/3 - quiesce running workloads"
  ( cd "$APP_DIR" && ./scripts/traffic_simulator.sh stop ) 2>/dev/null || true
  pkill -f "kubectl.*port-forward" 2>/dev/null || true
  ok "Background processes stopped."

  log "Step 2/3 - uninstall Docker Desktop"
  if [ ! -d "/Applications/Docker.app" ]; then
    warn "/Applications/Docker.app not found; nothing to uninstall."
  else
    confirm "About to uninstall Docker Desktop and wipe all images, containers and the local Kubernetes cluster. Proceed?" || {
      warn "Aborted by user."
      return 1
    }
    osascript -e 'quit app "Docker"' 2>/dev/null || true
    sleep 2
    if [ -x "/Applications/Docker.app/Contents/MacOS/uninstall" ]; then
      log "Running official uninstaller (may prompt for sudo)..."
      sudo /Applications/Docker.app/Contents/MacOS/uninstall || true
    else
      warn "Official uninstaller missing; removing app bundle manually."
      sudo rm -rf /Applications/Docker.app
    fi
  fi

  log "Step 3/3 - purge user-level Docker state"
  rm -rf "$HOME/Library/Group Containers/group.com.docker" \
         "$HOME/Library/Containers/com.docker.docker" \
         "$HOME/Library/Application Support/Docker Desktop" \
         "$HOME/.docker" 2>/dev/null || true
  ok "Docker Desktop fully removed."
}

cmd_install_check() {
  if [ -d "/Applications/Docker.app" ] && docker version >/dev/null 2>&1; then
    ok "Docker is already installed and running."
  else
    log "Opening Docker Desktop download page in your browser..."
    open "https://www.docker.com/products/docker-desktop/" || true
    cat <<EOF

  Manual steps in the browser/installer:
    1. Download the .dmg (Apple Silicon or Intel - match your chip).
    2. Drag Docker.app into /Applications.
    3. Launch Docker, sign in, choose "Personal use" if prompted.
    4. Settings -> Kubernetes -> [x] Enable Kubernetes -> Apply & Restart.
    5. Wait for the menubar whale icon to go SOLID (no orange dots).

EOF
    read -r -p "Press <Enter> once Docker Desktop is installed and running... " _
  fi

  wait_for_docker || return 1
  wait_for_k8s   || return 1
  ok "Docker + Kubernetes are healthy."
}

cmd_recover() {
  log "Recovering the heart-disease MLOps stack..."
  command -v kubectl >/dev/null || { err "kubectl missing - brew install kubectl"; return 1; }
  command -v helm    >/dev/null || { err "helm missing - brew install helm";       return 1; }

  ( cd "$APP_DIR" && ./scripts/setup_local_k8s.sh ) || { err "setup_local_k8s.sh failed"; return 1; }
  ( cd "$APP_DIR" && ./scripts/setup_monitoring.sh ) || { err "setup_monitoring.sh failed"; return 1; }

  log "Waiting for app + monitoring deployments..."
  kubectl -n heart-disease wait --for=condition=available deploy/heart-disease-api --timeout=180s || true
  kubectl -n monitoring   wait --for=condition=available deploy/grafana             --timeout=240s || true

  log "Re-establishing port-forwards (background)..."
  pkill -f "kubectl.*port-forward.*9090" 2>/dev/null || true
  pkill -f "kubectl.*port-forward.*3000" 2>/dev/null || true
  nohup kubectl -n monitoring port-forward svc/prometheus-server 9090:80 \
        >/tmp/pf-prom.log 2>&1 &
  nohup kubectl -n monitoring port-forward svc/grafana 3000:80 \
        >/tmp/pf-graf.log 2>&1 &
  disown -a 2>/dev/null || true

  log "Restarting traffic simulator..."
  ( cd "$APP_DIR" && ./scripts/traffic_simulator.sh start ) || true

  echo
  ok "Recovery complete. Verify:"
  echo "    curl http://localhost:8080/health"
  echo "    open http://localhost:9090   (Prometheus)"
  echo "    open http://localhost:3000   (Grafana, admin/admin)"
}

menu() {
  echo
  echo "========= Docker Recovery Menu ========="
  echo "  1) Uninstall Docker Desktop"
  echo "  2) Verify / install Docker Desktop"
  echo "  3) Recover MLOps stack (image + helm + port-forwards + traffic)"
  echo "  4) Full sequence (1 -> 2 -> 3)"
  echo "  q) Quit"
  echo "========================================"
  read -r -p "Choose: " choice
  case "$choice" in
    1) cmd_uninstall ;;
    2) cmd_install_check ;;
    3) cmd_recover ;;
    4) cmd_uninstall && cmd_install_check && cmd_recover ;;
    q|Q) exit 0 ;;
    *) err "Invalid choice"; menu ;;
  esac
}

case "${1:-menu}" in
  uninstall)     cmd_uninstall ;;
  install-check) cmd_install_check ;;
  recover)       cmd_recover ;;
  all)           cmd_uninstall && cmd_install_check && cmd_recover ;;
  menu|"")       menu ;;
  -h|--help)     sed -n '2,12p' "$0" ;;
  *) err "Unknown command: $1"; sed -n '2,12p' "$0"; exit 1 ;;
esac
