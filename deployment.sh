#! /bin/bash
sudo apt update -y
sudo apt upgrade -y
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update -y
sudo apt install -y python3.11 python3.11-venv python3.11-dev nodejs npm nginx certbot python3-certbot-nginx

python3.11 -m venv ~/test/venv
source ~/test/venv/bin/activate
pip install -r ~/test/Unthinkable/backend/requirements.txt
pip install gunicorn

cd ~/test/Unthinkable/frontend/
npm install
npm run build
cd ~/test

nginx_file_path="/etc/nginx/sites-available/codemos.conf"
sudo bash -c "cat > $nginx_file_path << 'EOF'
server {
    listen 80;
    server_name codemos-services.co.in www.codemos-services.co.in;
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl;
    server_name codemos-services.co.in www.codemos-services.co.in;

    root /home/ubuntu/test/Unthinkable/frontend/dist;
    index index.html;

    ssl_certificate /etc/letsencrypt/live/codemos-services.co.in/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/codemos-services.co.in/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    # Serve frontend files
    location / {
        try_files \$uri \$uri/ @backend; 
    }

    # Proxy API requests
    location @backend {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    access_log /var/log/nginx/codemos_access.log;
    error_log /var/log/nginx/codemos_error.log;
}
EOF"

sudo ln -sf /etc/nginx/sites-available/codemos.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx
sudo systemctl enable nginx

sudo certbot certonly --nginx -d codemos-services.co.in -d www.codemos-services.co.in --non-interactive --agree-tos -m admin@codemos-services.co.in

sudo mkdir -p /var/log/gunicorn
sudo chown -R ubuntu:www-data /var/log/gunicorn
sudo chmod -R 775 /var/log/gunicorn

gunicorn_file_path="/etc/systemd/system/gunicorn.service"
sudo bash -c "cat > $gunicorn_file_path << 'EOF'
[Unit]
Description=Gunicorn service for FastAPI (Smart Resume Screener)
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/test/Unthinkable/backend
Environment=\"PATH=/home/ubuntu/test/venv/bin\"
ExecStart=/home/ubuntu/test/venv/bin/gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  --bind 127.0.0.1:8000 \
  --workers 4 --timeout 120 \
  --access-logfile /var/log/gunicorn/access.log \
  --error-logfile /var/log/gunicorn/error.log

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF"

sudo systemctl daemon-reload
sudo systemctl enable gunicorn
sudo systemctl restart gunicorn
sudo systemctl restart nginx

sudo chown -R ubuntu:www-data /home/ubuntu/test/Unthinkable/frontend/dist
sudo chmod -R 755 /home/ubuntu/test/Unthinkable/frontend/dist

sudo chmod 755 /home/ubuntu
sudo chmod 755 /home/ubuntu/test
sudo chmod 755 /home/ubuntu/test/Unthinkable
sudo chmod 755 /home/ubuntu/test/Unthinkable/frontend

sudo systemctl restart nginx
sudo systemctl restart gunicorn
