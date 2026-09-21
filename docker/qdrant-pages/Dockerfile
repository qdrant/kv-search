# The binary is compiled on Linux by build.sh with the pinned source + patch.
# Ubuntu 24.04 matches the build host's glibc; no GPU runtime is required.
FROM ubuntu:24.04
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /qdrant
ARG SOURCE_REVISION
ARG PATCH_SHA256
LABEL org.opencontainers.image.title="Qdrant page-attention demo" \
      org.opencontainers.image.version="1.19.2-dev-page-attention" \
      org.opencontainers.image.description="Experimental immutable page-attention index with optional original-row rescoring" \
      org.opencontainers.image.source="https://github.com/qdrant/kv-search" \
      org.opencontainers.image.revision="${SOURCE_REVISION}" \
      io.qdrant.page-attention.patch-sha256="${PATCH_SHA256}"
COPY qdrant /qdrant/qdrant
COPY config.yaml /qdrant/config/config.yaml
COPY source.json build.json /qdrant/
COPY LICENSE NOTICE /qdrant/
ENV QDRANT__TELEMETRY_DISABLED=true RAYON_NUM_THREADS=12
EXPOSE 6333 6334
HEALTHCHECK --interval=5s --timeout=3s --start-period=30s \
    CMD curl --fail --silent http://127.0.0.1:6333/readyz || exit 1
ENTRYPOINT ["/qdrant/qdrant"]
