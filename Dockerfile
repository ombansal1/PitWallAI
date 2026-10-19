# PitWall AI - container image (used for AWS EC2, or any Docker host). Includes FastF1.
#   docker build -t pitwall-ai .
#   docker run -d -p 80:8501 -e GROQ_API_KEY=gsk_... pitwall-ai
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PITWALL_DECISION_LOG=/app/data/decisions/decision_log.jsonl

WORKDIR /app
COPY requirements.txt requirements-optional.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt -r requirements-optional.txt

COPY . .
# Retrain the ML model if the installed scikit-learn differs from the one that saved it (takes ~15 s).
RUN python -c "from pitwall.ml.tyre_model import load_model, train_and_save; load_model() or train_and_save()"

# Run as a normal user, not root. The two data folders are created here so the Docker volumes
# mounted on them are writable by that user.
RUN mkdir -p /app/data/decisions /app/data/fastf1_cache \
 && useradd --create-home pitwall && chown -R pitwall /app
USER pitwall

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=4)"
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
