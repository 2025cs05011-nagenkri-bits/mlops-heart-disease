#!/bin/bash
# traffic_simulator.sh
#
# Generates continuous traffic against the heart-disease-api so the
# Prometheus rate() queries on the Grafana dashboard have data.
#
# Usage:
#   ./scripts/traffic_simulator.sh start       # start (default URL http://localhost:8080)
#   ./scripts/traffic_simulator.sh start http://heart.example.com
#   ./scripts/traffic_simulator.sh stop
#   ./scripts/traffic_simulator.sh status
#   ./scripts/traffic_simulator.sh logs        # tail the request log
#
# Each iteration sends 1 GET /health + 2 POST /predict (one high-risk,
# one low-risk patient) every INTERVAL seconds (default 2s).

set -e

PIDFILE="/tmp/heart_traffic.pid"
LOGFILE="/tmp/heart_traffic.log"
DEFAULT_URL="http://localhost:8080"
INTERVAL="${INTERVAL:-2}"

start() {
  local url="${1:-$DEFAULT_URL}"

  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    echo "Already running (PID $(cat "$PIDFILE")). Use 'stop' first."
    exit 1
  fi

  echo "Starting traffic simulator..."
  echo "  Target  : $url"
  echo "  Interval: ${INTERVAL}s per cycle (1 health + 2 predicts = ~$(echo "3/$INTERVAL" | bc -l | xargs printf "%.1f") req/s)"
  echo "  PID file: $PIDFILE"
  echo "  Log file: $LOGFILE"

  nohup bash -c "
    while true; do
      ts=\$(date +%H:%M:%S)
      hc=\$(curl -s -o /dev/null -w '%{http_code}' '$url/health')
      p1=\$(curl -s -o /dev/null -w '%{http_code}' -X POST '$url/predict' \
        -H 'Content-Type: application/json' \
        -d '{\"age\":67,\"sex\":1,\"cp\":4,\"trestbps\":160,\"chol\":286,\"fbs\":0,\"restecg\":2,\"thalach\":108,\"exang\":1,\"oldpeak\":1.5,\"slope\":2,\"ca\":3,\"thal\":7}')
      p2=\$(curl -s -o /dev/null -w '%{http_code}' -X POST '$url/predict' \
        -H 'Content-Type: application/json' \
        -d '{\"age\":35,\"sex\":0,\"cp\":1,\"trestbps\":110,\"chol\":180,\"fbs\":0,\"restecg\":0,\"thalach\":170,\"exang\":0,\"oldpeak\":0.0,\"slope\":1,\"ca\":0,\"thal\":3}')
      echo \"\$ts  health=\$hc  predict_high=\$p1  predict_low=\$p2\"
      sleep $INTERVAL
    done
  " >"$LOGFILE" 2>&1 &
  PID=$!
  disown
  echo "$PID" > "$PIDFILE"

  sleep 1
  if kill -0 "$PID" 2>/dev/null; then
    echo "Started (PID $PID)."
    echo "Tail logs : ./scripts/traffic_simulator.sh logs"
    echo "Stop      : ./scripts/traffic_simulator.sh stop"
  else
    echo "Failed to start. Check $LOGFILE"
    rm -f "$PIDFILE"
    exit 1
  fi
}

stop() {
  if [ ! -f "$PIDFILE" ]; then
    echo "Not running (no PID file)."
    exit 0
  fi
  PID=$(cat "$PIDFILE")
  if kill -0 "$PID" 2>/dev/null; then
    pkill -P "$PID" 2>/dev/null || true
    kill "$PID" 2>/dev/null || true
    sleep 1
    echo "Stopped (was PID $PID)."
  else
    echo "PID $PID was already dead."
  fi
  rm -f "$PIDFILE"
}

status() {
  if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
    PID=$(cat "$PIDFILE")
    echo "Running (PID $PID)."
    echo "Last 5 cycles:"
    tail -5 "$LOGFILE" 2>/dev/null || echo "  (no log yet)"
  else
    echo "Not running."
    [ -f "$PIDFILE" ] && rm -f "$PIDFILE"
  fi
}

logs() {
  if [ ! -f "$LOGFILE" ]; then
    echo "No log file at $LOGFILE"
    exit 1
  fi
  tail -f "$LOGFILE"
}

case "${1:-}" in
  start)  start "${2:-}" ;;
  stop)   stop ;;
  status) status ;;
  logs)   logs ;;
  *)
    echo "Usage: $0 {start [url]|stop|status|logs}"
    echo ""
    echo "Examples:"
    echo "  $0 start                              # default http://localhost:8080"
    echo "  $0 start http://localhost:8080"
    echo "  INTERVAL=1 $0 start                   # faster (4.5 req/s)"
    echo "  $0 status"
    echo "  $0 logs"
    echo "  $0 stop"
    exit 1
    ;;
esac
