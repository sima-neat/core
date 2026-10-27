# CI provisioning only. Platform packages never enter the offline payload.
ARG CARD_IMAGE
FROM ${CARD_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
ADD sdk-tools.tar.gz /opt/offline-sdk-tools/
COPY payload /tmp/core-publication
COPY sysroot_baseline.py /opt/offline-sdk-tools/sysroot_baseline.py
RUN dpkg-query -W '-f=${binary:Package}\n' > /tmp/offline-baseline-packages \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-apt ca-certificates \
    && python3 /opt/offline-sdk-tools/sysroot_baseline.py /tmp/core-publication \
    && rm -rf /tmp/core-publication /tmp/offline-sysroot-packages /var/lib/apt/lists/*
