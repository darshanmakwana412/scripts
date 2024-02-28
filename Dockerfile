# Use the NVIDIA PyTorch image as the base
FROM nvcr.io/nvidia/pytorch:23.02-py3

# Argument for user ID
ARG USER_ID=1000
# Argument for group ID
ARG GROUP_ID=1000

# Create a group and user
RUN groupadd -g $GROUP_ID darshan && \
    useradd -l -u $USER_ID -g darshan darshan && \
    install -d -m 0755 -o darshan -g darshan /home/darshan

# Switch to the new user
USER darshan

# Set the working directory
WORKDIR /workspace