FROM us-central1-docker.pkg.dev/cloud-workstations-images/predefined/code-oss:latest
USER root
RUN apt-get update && \
    apt-get install -y \
    python3 \
    python3-pip \
    google-cloud-sdk
# Install dependencies by copying the requirements file first, for better layer caching
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -rf /root/.cache/pip
RUN pip install uv --no-cache-dir

# Copy application code
COPY . /app
WORKDIR /app

# Switch to a non-root user for security
USER user

# Expose the port the app runs on
EXPOSE 5002

# The command to run the application
CMD ["python", "run.py", "web"]
