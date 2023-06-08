FROM python:slim-buster
RUN apt-get update && \
    mkdir backend-project
WORKDIR /backend-project
ADD . backend-project
RUN pip install -r backend_requirements.txt
EXPOSE 8080
CMD ["python3", "routes.py"]