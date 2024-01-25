FROM python:3.8-slim

WORKDIR /app

# Copy the model directory and the Python script into the container
ADD model-best model-best
COPY predict.py /app/
COPY requirements.txt /app/requirements.txt

# Install any required dependencies
RUN pip install -r requirements.txt

EXPOSE 5001
RUN python -m spacy download en_core_web_sm

CMD ["python", "predict.py"]