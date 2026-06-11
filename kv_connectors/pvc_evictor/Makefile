IMAGE_REPOSITORY ?= quay.io/pvc-evictor/pvc-evictor
IMAGE_TAG ?= latest

.PHONY: test docker-build

test:
	python3 -m pytest tests/ -v

docker-build:
	cd .. && podman build -f pvc_evictor/Dockerfile -t $(IMAGE_REPOSITORY):$(IMAGE_TAG) .
