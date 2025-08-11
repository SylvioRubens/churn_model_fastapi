FROM python:3.12-slim-bullseye AS base

USER root

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

FROM base 

COPY . .

CMD [ "gunicorn" , "-c", "gunicorn.conf.py" ]