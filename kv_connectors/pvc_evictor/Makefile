.PHONY: test docker-build

test:
	python3 -m pytest tests/ -v

docker-build:
	cd .. && podman build -f pvc_evictor/Dockerfile -t pvc-evictor:filemapper-v0.21 .
