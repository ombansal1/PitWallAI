#!/bin/bash
# One-time setup on a fresh Amazon Linux 2023 EC2 instance (run as ec2-user):
#   bash ec2-setup.sh
set -euo pipefail
sudo dnf install -y docker unzip
sudo systemctl enable --now docker
sudo usermod -aG docker ec2-user
# docker compose plugin
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo curl -sSL "https://github.com/docker/compose/releases/latest/download/docker-compose-linux-$(uname -m)" \
  -o /usr/local/lib/docker/cli-plugins/docker-compose
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
# docker buildx plugin (docker compose build needs buildx 0.17 or later; Amazon Linux's is older)
BUILDX_VERSION=v0.19.3
ARCH=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/')
sudo curl -sSL "https://github.com/docker/buildx/releases/download/${BUILDX_VERSION}/buildx-${BUILDX_VERSION}.linux-${ARCH}" \
  -o /usr/local/lib/docker/cli-plugins/docker-buildx
sudo chmod +x /usr/local/lib/docker/cli-plugins/docker-buildx
# 1 GB swap so a 1 GB t3.micro has headroom while building the image
if [ ! -f /swapfile ]; then
  sudo dd if=/dev/zero of=/swapfile bs=1M count=1024 status=none
  sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
echo "Done. Log out and back in (so the docker group applies), then run deploy/aws/deploy.sh from the project folder."
