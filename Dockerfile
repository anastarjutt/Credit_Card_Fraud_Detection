FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /project

# Copy all project files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python","main.py","Train","--model","xgb" ]