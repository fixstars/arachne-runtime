.PHONY: devel ubuntu1804-gpu ubuntu2004-gpu jp45 jp46

devel:
	docker build -t arachne-runtime:devel \
	-f devel.Dockerfile .

ubuntu1804-gpu:
	docker build -t arachne-runtime:ubuntu1804-gpu \
	-f ubuntu1804-gpu.Dockerfile .

jp46:
	docker build -t arachne-runtime:jp46 \
	-f jp46.Dockerfile .

jp45:
	docker build -t arachne-runtime:jp45 \
	-f jp45.Dockerfile .
