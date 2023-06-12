FROM python:3.7.17-slim-buster

WORKDIR /app

RUN apt-get update

COPY frontend_reqirements.txt .
RUN pip install -r frontend_reqirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "home_page.py", "--server.port=8501", "--server.address=0.0.0.0" ]