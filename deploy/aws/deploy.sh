#!/bin/bash
# Build and (re)start PitWall AI on the server. Run from the project folder: bash deploy/aws/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
docker compose up -d --build
docker image prune -f >/dev/null
docker builder prune -af --filter "until=72h" >/dev/null 2>&1 || true   # keeps the 16 GB disk from filling up
IP=$(curl -s -m 3 http://checkip.amazonaws.com || echo "<your-public-ip>")
echo "PitWall AI is starting. Open http://${IP}/ in a minute."
