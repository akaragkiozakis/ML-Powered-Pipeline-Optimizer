# -----------------------------
# 1. Base image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# 2. Working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# 3. Copy project files
# -----------------------------
COPY . /app

# -----------------------------
# 4. Install dependencies
# -----------------------------
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# 5. Expose FastAPI port
# -----------------------------
EXPOSE 8000

# -----------------------------
# 6. Run FastAPI with Uvicorn
# -----------------------------
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
