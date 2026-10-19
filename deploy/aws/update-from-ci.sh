#!/bin/bash
# Runs ON THE SERVER, copied to /tmp and started by the GitHub Actions workflow (.github/workflows/deploy.yml).
# Swaps in the new code from /tmp/pitwall_ai.tgz, keeps the server's own .env (the AI key),
# then rebuilds and restarts the app. Decision records and FastF1 downloads live in Docker
# volumes, so they are kept.
set -euo pipefail
APP="$HOME/pitwall_ai"
NEW="$HOME/pitwall_ai.new"
OLD="$HOME/pitwall_ai.old"

rm -rf "$NEW" && mkdir -p "$NEW"
tar xzf /tmp/pitwall_ai.tgz -C "$NEW"
rm -f /tmp/pitwall_ai.tgz

if [ -f "$APP/.env" ]; then
  cp "$APP/.env" "$NEW/.env"
else
  echo "Note: no $APP/.env on the server, so the agents will use their rules (no AI key)."
fi

rm -rf "$OLD"
if [ -d "$APP" ]; then mv "$APP" "$OLD"; fi   # the previous version is kept in ~/pitwall_ai.old
mv "$NEW" "$APP"

cd "$APP"
bash deploy/aws/deploy.sh

# Wait until the app answers (up to about 3 minutes).
for i in $(seq 1 36); do
  if curl -fsS -m 4 http://localhost/_stcore/health >/dev/null 2>&1; then
    echo "PitWall AI is up."
    exit 0
  fi
  sleep 5
done
echo "The app did not answer in time. Last log lines:"
docker compose logs --tail 40
exit 1
