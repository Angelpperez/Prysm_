FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY api_service ./api_service
EXPOSE 8000
ENV PYTHONUNBUFFERED=1
ENV PRYSM_VISION_SERVICE_URL=http://vision:8001
CMD python -m api_service.main
