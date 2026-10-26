FROM python:3.12

# Create a non-root user
ARG UID
ARG MYUSER
RUN useradd -u $UID --create-home ${MYUSER}

# Switch to the non-root user to set up the workspace
USER ${MYUSER}
WORKDIR /home/${MYUSER}/
COPY --chown=${MYUSER} --chmod=765 . .

# Switch back to root for installation steps
USER root

# Install required system packages
RUN apt-get update && apt-get install -y tmux

# Install the package and its optional dependencies (non‑editable install)
RUN pip install -e .

USER ${MYUSER}
