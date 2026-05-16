FROM python:3.10-slim

WORKDIR /api

COPY backend/ ./backend/
COPY requirements.txt .
COPY /model/artifact/ ./model/artifact/

RUN pip install -r requirements.txt

EXPOSE 7860

CMD [ "uvicorn","backend.api:app","--host", "0.0.0.0", "--port", "7860"]


