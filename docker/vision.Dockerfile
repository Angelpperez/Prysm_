FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY vision_service ./vision_service
COPY vision_app ./vision_app
EXPOSE 8001
ENV PYTHONUNBUFFERED=1
CMD python -m vision_service.main
