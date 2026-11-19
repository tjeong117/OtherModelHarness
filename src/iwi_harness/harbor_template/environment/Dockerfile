RUN apt-get update && apt-get install -y --no-install-recommends tmux asciinema \
 && rm -rf /var/lib/apt/lists/*
COPY iwi_harness /opt/iwi/iwi_harness
COPY edit-target /usr/local/bin/edit-target
RUN chmod +x /usr/local/bin/edit-target && mkdir -p /logs/artifacts
ENV PYTHONPATH=/opt/iwi
WORKDIR /work
