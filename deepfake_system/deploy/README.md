# Deployment — Ubuntu 22.04 LTS

Matches the stack in IR chapter 2.4: Ubuntu 22.04, Nginx in front,
DynamoDB for stored findings.

```bash
sudo apt update && sudo apt install -y python3-venv ffmpeg nginx
sudo useradd -r -s /usr/sbin/nologin deepfake
sudo mkdir -p /opt/deepfake_system && sudo chown deepfake: /opt/deepfake_system
sudo -u deepfake cp -r ./deepfake_system/* /opt/deepfake_system/

cd /opt/deepfake_system
sudo -u deepfake python3 -m venv .venv
sudo -u deepfake .venv/bin/pip install -r requirements-web.txt
sudo -u deepfake mkdir -p reports

sudo cp deploy/deepfake.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now deepfake

sudo cp deploy/nginx.conf /etc/nginx/sites-available/deepfake
sudo ln -s /etc/nginx/sites-available/deepfake /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo certbot --nginx -d your-domain
sudo nginx -t && sudo systemctl reload nginx
```

## DynamoDB table

```bash
aws dynamodb create-table \
  --table-name deepfake_reports \
  --attribute-definitions AttributeName=report_id,AttributeType=S \
                          AttributeName=created_at,AttributeType=S \
  --key-schema AttributeName=report_id,KeyType=HASH \
               AttributeName=created_at,KeyType=RANGE \
  --billing-mode PAY_PER_REQUEST \
  --region ap-southeast-1
```

Attach an IAM role to the instance with `dynamodb:PutItem`, `GetItem`
and `Scan` on that table and nothing else. Do not put access keys in the
repository — the report will be marked down for it and rightly so.

Without credentials the app writes to `reports/reports.jsonl` instead and
says so at `/api/status`. That is the intended local-development path,
not a failure.

## Checking the security requirements hold

```bash
curl -sI https://your-domain | grep -i strict-transport   # HSTS present
curl -sI http://your-domain | head -1                     # 301 to HTTPS
sudo ls -la /tmp                                          # no leftover uploads
sudo journalctl -u deepfake -n 50                         # no video paths logged
```

The uploaded file is deleted in a `finally` block in `app/server.py`
whether the analysis succeeds or throws. `PrivateTmp=true` in the unit
gives the service its own `/tmp` that is destroyed when it stops, so
even a crash mid-analysis leaves nothing behind.
