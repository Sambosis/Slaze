FROM python:3.12-slim

# Install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm -rf /root/.cache/pip

# Copy application code
COPY . /app
WORKDIR /app

EXPOSE 5000
CMD ["python", "run.py", "web"]
