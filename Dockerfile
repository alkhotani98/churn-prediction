FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install system dependencies (for pandas, numpy, pyarrow, etc.)
RUN apt-get update && apt-get install -y build-essential

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose API port
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]