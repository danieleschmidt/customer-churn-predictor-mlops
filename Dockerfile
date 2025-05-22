# Start with a Python base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies
# Add any system dependencies needed by your packages here (e.g., for opencv, etc.)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This assumes your Dockerfile is in the root of your project
# and your application code (src, models, etc.) is also in the root.
COPY . .
# If your model is very large, consider adding it to .dockerignore if you build it elsewhere
# and only copy the necessary src/ and models/churn_model.joblib

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable for the port (optional, Flask script can use it)
ENV PORT 5000

# Command to run the application
# This will run src/api.py using python.
# Ensure api.py is executable or called via python module.
CMD ["python", "src/api.py"]
