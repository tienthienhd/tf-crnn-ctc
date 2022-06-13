#!/bin/bash
docker build -t  docker-registry.tima-ai.dev/thiennt/captcha_service:1.0.5 .
docker push  docker-registry.tima-ai.dev/thiennt/captcha_service:1.0.5