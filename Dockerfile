FROM us-central1-docker.pkg.dev/cloud-workstations-images/predefined/code-oss:latest
USER root
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    google-cloud-sdk
RUN git clone http://github.com/sambosis/Slaze /app
# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -rf /root/.cache/pip
RUN pip install uv --no-cache-dir

# Copy application code

WORKDIR /app

EXPOSE 5002
# CMD ["python", "run.py", "web"]
USER user
