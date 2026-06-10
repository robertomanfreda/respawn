ARG PYTORCH_IMAGE=nvcr.io/nvidia/pytorch:25.12-py3
FROM ${PYTORCH_IMAGE}

ARG COMFYUI_REPOSITORY=https://github.com/Comfy-Org/ComfyUI.git
ARG COMFYUI_REF=master

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone --depth 1 --branch "${COMFYUI_REF}" "${COMFYUI_REPOSITORY}" /opt/ComfyUI \
    || (git clone "${COMFYUI_REPOSITORY}" /opt/ComfyUI && cd /opt/ComfyUI && git checkout "${COMFYUI_REF}")

COPY infra/docker/comfyui/patch_comfyui.py /tmp/respawn-patch-comfyui.py

WORKDIR /opt/ComfyUI
RUN python -m pip install --upgrade pip \
    && python /tmp/respawn-patch-comfyui.py requirements \
    && python -m pip install -r /tmp/comfyui-requirements-no-torch.txt \
    && python /tmp/respawn-patch-comfyui.py source

COPY infra/docker/comfyui/entrypoint.sh /usr/local/bin/respawn-comfyui-entrypoint
RUN chmod +x /usr/local/bin/respawn-comfyui-entrypoint

EXPOSE 8188
ENTRYPOINT ["respawn-comfyui-entrypoint"]
