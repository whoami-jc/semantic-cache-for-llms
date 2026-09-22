FROM python:3.12-slim

WORKDIR /service
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir sentence-transformers==3.4.1 transformers==4.48.3 fastapi==0.115.14 uvicorn==0.30.6
COPY server.py .
ENV HF_HOME=/models
EXPOSE 8001
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
