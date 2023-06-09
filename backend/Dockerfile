
FROM python:3.7.17-slim-buster

WORKDIR /app

RUN apt-get update

COPY backend_requirements.txt .
RUN pip install -r backend_requirements.txt

COPY . .

#EXPOSE 8080

CMD ["uvicorn","routes:app","--host", "0.0.0.0", "--port", "80","--reload"]