# Only the producer's Platform userspace library/headers enter the clean image.
ARG UBUNTU_IMAGE
FROM ${UBUNTU_IMAGE} AS platform-build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake python3 binutils
WORKDIR /input
COPY platform_provider.py platform-source.json platform-source.tar.gz ./
RUN python3 -c "import platform_provider as p; p.build('platform-source.tar.gz', '/platform', '/work')"
FROM ${UBUNTU_IMAGE}
COPY --from=platform-build /platform/ /
RUN ldconfig
