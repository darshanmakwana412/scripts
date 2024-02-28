#!/bin/bash

echo "Starting Docker container setup..."

echo "Current GPU status:"
nvidia-smi

# Default values
default_port=8888
current_user=$(whoami)
# Use the name of the current directory as the default container name
dir_name=$(basename "$(pwd)")
default_container_name="${dir_name}-container"
default_image_name="pytorch:23.02-py3"
settings_dir="docker"

# Create directory for docker settings if it doesn't exist
if [ ! -d "$settings_dir" ]; then
    mkdir -p $settings_dir
fi

is_port_available() {
    ! netstat -tuln | grep -q ":$1 "
}

find_next_available_port() {
    local port=$1
    while ! is_port_available $port; do
        ((port++))
    done
    echo $port
}

read -p "Enter a name for the new container (default: $default_container_name): " container_name
container_name=${container_name:-$default_container_name}

# Ask for custom Docker image or use default
read -p "Enter the Docker image to use (default: $default_image_name): " image_name
image_name=${image_name:-$default_image_name}

available_gpus=$(nvidia-smi --query-gpu=index --format=csv,noheader,nounits | paste -sd "," -)

# Check if any GPUs are available
if [ -z "$available_gpus" ]; then
    echo "No GPUs available."
    exit 1
fi

echo "All GPUs available: $available_gpus"

# Ask for custom port or find next available port starting from the default
read -p "Enter the port to use (default: $default_port): " port
port=${port:-$default_port}
port=$(find_next_available_port $port)

echo "Using port $port for the Docker container"

# Command to run Docker container, mapping user ID and group ID to match the host, enabling file editing from outside
docker_command="docker run -it --rm --detach-keys=\"ctrl-a\" --name=$container_name --gpus '\"device=$available_gpus\"' --ipc=host -p $port:8888 -v `pwd`:/workspace -u $(id -u):$(id -g) $image_name"

# Run Docker container
echo "Running Docker container with the following settings:"
echo "Container Name: $container_name"
echo "Selected GPUs: $available_gpus"
echo "Port: $port"
echo "Command running: $docker_command"

# Save settings to a new file in the specified directory
settings_file="${settings_dir}/docker_${container_name}.sh"
echo "#!/bin/bash" > $settings_file
echo $docker_command >> $settings_file
chmod +x $settings_file

eval $docker_command