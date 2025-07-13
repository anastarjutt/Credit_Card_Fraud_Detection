FROM python:3.11-slim

ENV PYTHONPATH=/project/src:$PYTHONPATH

WORKDIR /project

COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD [ "uvicorn","main:app","--host","0.0.0.0","--port","8000","--reload","--workers","1","--log-loss","info","--python-path","project/src" ]