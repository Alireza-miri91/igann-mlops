FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.13.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/ models/
COPY data/ data/

EXPOSE 8000

CMD ["uvicorn","src.serve:app","--host","0.0.0.0","--port","8000"]