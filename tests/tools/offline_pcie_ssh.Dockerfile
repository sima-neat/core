# Disposable test fixture only; not a shipped runtime image.
FROM ubuntu:24.04
RUN apt-get update -qq \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
       python3 openssh-server iproute2 \
    && rm -rf /var/lib/apt/lists/*
