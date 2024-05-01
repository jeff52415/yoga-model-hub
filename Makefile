# Makefile for managing the FastAPI application

# Define variables
IMAGE_NAME=yoga-hub
CONTAINER_NAME=yoga-hub
PORT=8000
TAG=cpu-v1.1.0


# Build the Docker image
build:
	docker build -f docker/DockerfileCPU -t $(IMAGE_NAME):$(TAG) .

# Run the Docker container
run:
	docker run -d --name $(CONTAINER_NAME) -p $(PORT):$(PORT) $(IMAGE_NAME):$(TAG)

# Stop the Docker container
stop:
	docker stop $(CONTAINER_NAME)

# Remove the Docker container
remove: stop
	docker rm $(CONTAINER_NAME)

# Access the container's shell
shell:
	docker exec -it $(CONTAINER_NAME) /bin/sh

# View container logs
logs:
	docker logs $(CONTAINER_NAME)

# Run tests (you can customize this according to your test setup)
test:
	python -m unittest discover -s tests

# Clean up (add any additional clean-up commands as needed)
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

lint:
	pre-commit run --all-files

test-api:
	curl -X POST "http://localhost:8000/predict" \
		-H "accept: application/json" \
		-H "Content-Type: multipart/form-data" \
		-F "image=@example/test.png"

.PHONY: build run stop remove shell logs test clean
