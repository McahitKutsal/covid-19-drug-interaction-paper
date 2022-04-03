FROM python:3.9.6

ADD . /app
WORKDIR /app

RUN pip install -r requirements.txt
RUN python main.py