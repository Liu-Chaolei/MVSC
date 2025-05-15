# FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel
FROM nvcr.io/nvidia/pytorch:24.08-py3
ARG DEBIAN_FRONTEND=noninteractive
RUN set -x


###### system-level dependencies ######
RUN apt-get update && \
    apt-get install -y git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


###### install cogkit ######
WORKDIR /app

# refactor url later (and maybe repo name)
RUN git clone https://github.com/THUDM/CogKit.git
WORKDIR CogKit

# TODO: use `pdm sync`
RUN pip install uv
RUN uv pip install . --system
