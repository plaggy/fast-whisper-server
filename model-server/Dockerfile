FROM nvidia/cuda:12.3.2-devel-ubuntu22.04

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libncurses5-dev \
    zlib1g-dev \
    libreadline-dev \
    libbz2-dev \
    libsqlite3-dev \
    wget \
    ffmpeg \
    git \
    && apt-get clean autoremove --yes \
    && rm -rf /var/lib/{apt,dpkg,cache,log}

ENV HOME="/root"
WORKDIR ${HOME}

RUN git clone --depth=1 https://github.com/pyenv/pyenv.git .pyenv

ENV PYENV_ROOT="${HOME}/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"

RUN eval "$(pyenv init -)"

RUN pyenv install 3.10
RUN pyenv global 3.10

RUN pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install flash-attn --no-build-isolation

WORKDIR /usr/src

COPY app app

EXPOSE 7860

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--log-config", "app/utils/log_config.yaml"]