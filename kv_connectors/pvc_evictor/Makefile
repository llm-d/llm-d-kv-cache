IMAGE_REPOSITORY ?= quay.io/pvc-evictor/pvc-evictor
IMAGE_TAG ?= llm-d-v0.8

.PHONY: test docker-build

test:
	python3 -m pytest tests/ -v

docker-build:
	cd .. && podman build -f pvc_evictor/Dockerfile -t $(IMAGE_REPOSITORY):$(IMAGE_TAG) .
