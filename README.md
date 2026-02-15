# Prysm_

Two-service layout:

- vision_service: owns camera, detection, calibration, PLC, and MJPEG stream.
- api_service: UI (FastHTML) and API proxy.

## Local run
pip install -r requirements.txt
python -m vision_service.main
python -m api_service.main

## Docker (Windows: API only)
Run vision_service on the host:
python -m vision_service.main

Then run api_service in Docker:
docker compose up --build

## Stream proxy toggle
Set PRYSM_USE_STREAM_PROXY=false to let the browser connect directly to vision_service.

## Docker on Windows (no USB)
Use the synthetic camera source:
docker compose -f docker-compose.yml -f docker-compose.windows.yml up --build

## Endpoints
vision_service
- GET /health
- GET /status
- GET /frame
- GET /stream
- POST /calibrate

api_service
- GET /
- GET /status
- GET /api/status
- POST /api/calibrate

## Env
See config/env.example

Note: set PRYSM_VISION_PUBLIC_URL to a host-reachable URL for the browser (for example http://localhost:8001).



Para encender
 python -m vision_service.main
docker compose logs -f vision
