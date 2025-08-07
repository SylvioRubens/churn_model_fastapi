# gunicorn.conf.py

import multiprocessing
import os


wsgi_app = "app.main:app"

# Número de workers (ex: cores x 2 + 1)
# workers = multiprocessing.cpu_count() * 2 + 1
workers = 2 

# Worker class: Uvicorn async worker
worker_class = "uvicorn.workers.UvicornWorker"

# Bind address
bind = "0.0.0.0:8000"

# Logging
accesslog = "logs/access.log"
errorlog = "logs/error.log"
loglevel = "info"

# Tempo máximo por requisição
timeout = 60
keepalive = 5

reload = True if os.getenv("ENVIRONMENT") == "dev" else False