sudo apt update
sudo apt upgrade
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
sudo apt install nodejs
sudo apt install npm

mkdir test
cd test
git clone https://github.com/kshitijdalvi4/Unthinkable.git

#virtual environemnt
python3.11 -m venv venv
source venv/bin/activate

#install python packages
pip install -r ~/test/Unthinkable/backend/requirements.txt

#install frontend packages and build
cd ~/test/Unthinkable/frontend/
npm install
npm run build
cd ~/test

sudo apt update
sudo apt upgrade
sudo apt install nginx
nginx_file_path = "/etc/nginx/sites-available/codemos.conf"
sudo touch nginx_file_path
cat <<EOF > nginx_file_path 
    server {
    server_name codemos-services.co.in www.codemos-services.co.in;

    root /home/ubuntu/test/Unthinkable/frontend/dist;
    index index.html;

    location / {
        # If the requested file exists (like JS, CSS, images), serve it
        try_files $uri $uri/ @backend;
    }

    location @backend {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    access_log /var/log/nginx/codemos_access.log;
    error_log /var/log/nginx/codemos_error.log;

    listen 443 ssl; # managed by Certbot
    ssl_certificate /etc/letsencrypt/live/codemos-services.co.in/fullchain.pem; # managed by Certbot
    ssl_certificate_key /etc/letsencrypt/live/codemos-services.co.in/privkey.pem; # managed by Certbot
    include /etc/letsencrypt/options-ssl-nginx.conf; # managed by Certbot
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem; # managed by Certbot


}

server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name codemos-services.co.in www.codemos-services.co.in;

    ssl_certificate  /etc/letsencrypt/live/codemos-services.co.in/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/codemos-services.co.in/privkey.pem;
    include /etc/letsencrypt/options-ssl-nginx.conf;
    ssl_dhparam /etc/letsencrypt/ssl-dhparams.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}


server {
    if ($host = www.codemos-services.co.in) {
        return 301 https://$host$request_uri;
    } # managed by Certbot


    if ($host = codemos-services.co.in) {
        return 301 https://$host$request_uri;
    } # managed by Certbot


    listen 80;
    server_name codemos-services.co.in www.codemos-services.co.in;
    return 404; # managed by Certbot




}
EOF

sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable nginx
sudo systemctl start nginx
sudo systemctl status nginx

pip install gunicorn
gunicorn_file_path = /etc/systemd/system/gunicorn.service
sudo touch gunicorn_file_path

cat <<EOF > gunicorn_file_path
[Unit]
Description=Gunicorn service for FastAPI (Smart Resume Screener)
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/test/Unthinkable/backend
Environment="PATH=/home/ubuntu/test/venv/bin"
ExecStart=/home/ubuntu/test/venv/bin/gunicorn main:app \
  -k uvicorn.workers.UvicornWorker \
  --bind 127.0.0.1:8000 \
  --workers 4 --timeout 120 \
  --access-logfile /var/log/gunicorn/codemos_access.log \
  --error-logfile /var/log/gunicorn/codemos_error.log

Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable gunicorn.service
sudo systemctl start gunicorn.service
sudo systemctl status gunicorn.service
