#!/bin/bash

echo "Fetching running Docker containers..."

# Get a list of running containers (container ID and names)
containers=$(docker ps --format "{{.ID}} - {{.Names}}")

# Check if there are any running containers
if [ -z "$containers" ]; then
    echo "No running Docker containers found."
fi

# Calculate the number of scripts, but limit the maximum height
script_count=$(echo "$containers" | wc -l)
max_height=10
height=$((script_count+4)) # Add a little padding for fzf's UI elements
if [ $height -gt $max_height ]; then
    height=$max_height
fi

# Use fzf to let the user select a container
selected_container=$(echo "$containers" | fzf --height=$height --header="Select a container to attach to:")

# Extract the container ID from the selection
container_id=$(echo $selected_container | awk '{print $1}')

# Check if a container was selected
if [ -z "$container_id" ]; then
    echo "No container selected."
fi

echo "Attaching to container $container_id..."
docker attach --detach-keys="ctrl-a" $container_id