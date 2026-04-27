#!/bin/bash
# Degas Clips — Hetzner server setup script
# Run as root on a fresh Ubuntu 24.04 server

set -e

echo "=== Updating system ==="
apt-get update && apt-get upgrade -y

echo "=== Installing dependencies ==="
apt-get install -y python3 python3-pip python3-venv git ffmpeg

echo "=== Mounting data volume ==="
# Format and mount the Hetzner volume (run once only)
# Replace /dev/sdb with your actual volume device (check with: lsblk)
# mkfs.ext4 /dev/sdb          # Uncomment only on first run
# mkdir -p /data
# mount /dev/sdb /data
# echo "/dev/sdb /data ext4 defaults 0 2" >> /etc/fstab

echo "=== Creating data directories ==="
mkdir -p /data/uploads
mkdir -p /data/outputs
chmod 755 /data

echo "=== Cloning app ==="
cd /opt
git clone https://github.com/KaiserMediaHub/DegasClips.git degas
cd /opt/degas

echo "=== Creating virtual environment ==="
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Creating .env file ==="
cat > /opt/degas/.env << 'EOF'
SECRET_KEY=REPLACE_WITH_RANDOM_STRING
APP_PASSWORD=REPLACE_WITH_YOUR_PASSWORD
UPLOAD_FOLDER=/data/uploads
OUTPUT_FOLDER=/data/outputs
DB_PATH=/data/degas.db
MAX_CONTENT_MB=3072
EOF

echo "=== Installing systemd service ==="
cp /opt/degas/degas.service /etc/systemd/system/degas.service
systemctl daemon-reload
systemctl enable degas
systemctl start degas

echo ""
echo "=== Done! ==="
echo "Edit /opt/degas/.env to set your SECRET_KEY and APP_PASSWORD"
echo "Then run: systemctl restart degas"
