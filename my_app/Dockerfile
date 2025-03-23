FROM pytorch/pytorch:latest
WORKDIR /app
COPY requirements.txt .
COPY app.py .
RUN pip install --upgrade pip
RUN python3 -m pip install --user --no-cache-dir -r requirements.txt
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]
