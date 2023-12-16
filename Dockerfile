# Use an official miniconda image as a parent image
FROM python:3.10

# Set the working directory in docker
WORKDIR /app

# Create a directory for the data volume
COPY requirements.txt .

# Copy the Python script into the container at /app
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Always use the Python script as the entry point
ENTRYPOINT ["python", "src/recommendation/pipeline.py"]

# By default, write "hello world" to the file.
CMD ["--config", "config/dataset_V2.yaml", "--threshold", "0.75", "--model", "greedy", "-k", "2", "--total_steps", "5000", "--eval_freq", "500", "--nb_runs", "3"]