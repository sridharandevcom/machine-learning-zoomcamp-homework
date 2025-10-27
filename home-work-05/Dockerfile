FROM agrigorev/zoomcamp-model:2025

WORKDIR /app
COPY . .


RUN pip install uv

RUN uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install --system -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
