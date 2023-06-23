#!/bin/bash
#docker build -t  docker-registry.tima-ai.dev/thiennt/captcha_service:1.0.8 .
#docker push  docker-registry.tima-ai.dev/thiennt/captcha_service:1.0.8

docker build -t  tienthienhd/captcha_service:1.0.1 .
docker push  tienthienhd/captcha_service:1.0.1