FROM godatadriven/pyspark

# Copy the Python application code and the trained model into the container
COPY . /app

# Set the working directory to /app
WORKDIR /app

# Install any necessary dependencies
RUN pip install -r requirements.txt


# Set the entrypoint to the Python application
ENTRYPOINT ["python", "app.py"]
