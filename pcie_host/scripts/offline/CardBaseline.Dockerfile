# Build tools stay in the SDK stage; only resolved packages enter the clean card.
ARG SDK_IMAGE
ARG CARD_IMAGE
FROM ${SDK_IMAGE} AS sdk
ADD sdk-tools.tar.gz /opt/offline-sdk-tools/
COPY payload /tmp/core-publication
COPY sysroot_baseline.py /opt/offline-sdk-tools/sysroot_baseline.py
RUN python3 /opt/offline-sdk-tools/sysroot_baseline.py prepare /tmp/core-publication

FROM ${CARD_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive
COPY --from=sdk /tmp/offline-sysroot-packages /tmp/offline-sysroot-packages
COPY --from=sdk /var/lib/neat-offline-ci /var/lib/neat-offline-ci
COPY payload /tmp/core-publication
COPY sysroot_baseline.py /opt/offline-sdk-tools/sysroot_baseline.py
RUN dpkg-query -W '-f=${binary:Package}\n' > /tmp/offline-baseline-packages \
    && apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-apt ca-certificates \
    && python3 /opt/offline-sdk-tools/sysroot_baseline.py install /tmp/core-publication \
    && rm -rf /tmp/core-publication /tmp/offline-sysroot-packages /var/lib/apt/lists/*
