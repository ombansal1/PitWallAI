#!/bin/bash
# Build and (re)start PitWall AI on the server. Run from the project folder: bash deploy/aws/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
docker compose up -d --build
docker image prune -f >/dev/null
IP=$(curl -s -m 3 http://checkip.amazonaws.com || echo "<your-public-ip>")
echo "PitWall AI is starting. Open http://${IP}/ in a minute."
