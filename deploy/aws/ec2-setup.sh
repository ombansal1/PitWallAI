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
# 1 GB swap so a 1 GB t3.micro has headroom while building the image
if [ ! -f /swapfile ]; then
  sudo dd if=/dev/zero of=/swapfile bs=1M count=1024 status=none
  sudo chmod 600 /swapfile && sudo mkswap /swapfile >/dev/null && sudo swapon /swapfile
  echo '/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab >/dev/null
fi
echo "Done. Log out and back in (so the docker group applies), then run deploy/aws/deploy.sh from the project folder."
