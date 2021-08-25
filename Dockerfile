FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
ENV PORT 15000
ENV APP_MODULE app.api:app
ENV LOG_LEVEL debug
ENV WEB_CONCURRENCY 1

COPY ./app/api_requirements.txt ./requirements/api_requirements.txt
RUN pip install -r requirements/api_requirements.txt

COPY ./app /app/app
COPY ./config.py /app/app/config.py

#CMD ["ls", "-la" , "app"]