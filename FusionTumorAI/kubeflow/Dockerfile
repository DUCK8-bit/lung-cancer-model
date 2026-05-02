# Use a PyTorch base image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory inside the container
WORKDIR /app



# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Set the Python path so modules can be imported correctly
ENV PYTHONPATH="/app"

# Default command (can be overridden by Kubeflow components)
CMD ["python", "main.py"]
