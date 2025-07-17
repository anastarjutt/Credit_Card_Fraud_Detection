FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /project

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Allow CMD to be overridden by the arguments passed to docker run
ENTRYPOINT ["python", "main.py"]

# Default CMD (can be overridden by docker run arguments)
CMD ["Train", "--model", "xgb", "--tuner", "randomized"]
