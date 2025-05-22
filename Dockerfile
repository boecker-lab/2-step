FROM mambaorg/micromamba:latest

COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes

WORKDIR /app
COPY *.py /app
COPY models /app/models
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # otherwise python will not be found
