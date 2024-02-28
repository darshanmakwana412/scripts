# Custom image name
custom_image_name="pytorch:23.02-py3"

# Build the custom Docker image with the current user's UID and GID
docker build --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t $custom_image_name .