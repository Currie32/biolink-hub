FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy service code
COPY . .

# Download and index dictionaries (can be pre-built in CI)
# RUN python -m bioextract.dictionaries.download
# RUN python -m bioextract.dictionaries.index

EXPOSE 8001

CMD ["uvicorn", "bioextract.api:app", "--host", "0.0.0.0", "--port", "8001"]
