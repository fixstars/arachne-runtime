.PHONY: devel ci ubuntu1804-gpu ubuntu2004-gpu jp45 jp46 all

NO_CACHE = --no-cache

devel:
	docker build -t arachne-runtime:devel \
	-f devel.Dockerfile ${NO_CACHE} .
ci:
	docker build -t arachne-runtime:ci \
	-f ci.Dockerfile ${NO_CACHE} ../

ubuntu1804-gpu:
	docker build -t arachne-runtime:ubuntu1804-gpu \
	-f ubuntu1804-gpu.Dockerfile ${NO_CACHE} .

jp46:
	docker build -t arachne-runtime:jp46 \
	-f jp46.Dockerfile ${NO_CACHE} .

jp45:
	docker build -t arachne-runtime:jp45 \
	-f jp45.Dockerfile ${NO_CACHE} .

all: devel ci ubuntu1804 jp46 jp45
