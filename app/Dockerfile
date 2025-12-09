FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
