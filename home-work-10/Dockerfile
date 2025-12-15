FROM python:3.13-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY predict.py .
COPY model.bin .

EXPOSE 9696

CMD ["python", "predict.py"]