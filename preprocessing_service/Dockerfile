FROM python:3.8-slim

WORKDIR /app

# Copy the model directory and the Python script into the container
COPY preprocessing.py /app/
COPY requirements.txt /app/requirements.txt

# Install any required dependencies
RUN pip install -r requirements.txt

EXPOSE 5000

RUN python -m spacy download en_core_web_sm
CMD ["python", "preprocessing.py"]