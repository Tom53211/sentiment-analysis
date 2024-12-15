FROM python:3.9-slim

COPY src/app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000
ENV PYTHONPATH="$PYTHONPATH:$PWD"

CMD ["python3", "src/app/main.py"]
