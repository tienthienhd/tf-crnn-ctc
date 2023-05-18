FROM --platform=linux/amd64 tiangolo/uvicorn-gunicorn-fastapi:python3.7
ENV PORT 15000
ENV APP_MODULE app.api:app
ENV LOG_LEVEL debug
ENV WEB_CONCURRENCY 1

COPY api_requirements.txt /tmp/api_requirements.txt
RUN pip install -r /tmp/api_requirements.txt

COPY ./app /app/app
COPY ./config.py /app/app/config.py

#CMD ["ls", "-la" , "app"]