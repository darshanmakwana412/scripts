#!/bin/bash

echo "Fetching running Docker containers..."

# Get a list of running containers (container ID and names)
containers=$(docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Image}}" | tail -n +2)

# Check if there are any running containers
if [ -z "$containers" ]; then
    echo "No running Docker containers found."
    exit 1
fi

# Calculate the number of scripts, but limit the maximum height
script_count=$(echo "$containers" | wc -l)
max_height=10
height=$((script_count+4)) # Add a little padding for fzf's UI elements
if [ $height -gt $max_height ]; then
    height=$max_height
fi

# Use fzf to let the user select multiple containers
selected_containers=$(echo "$containers" | fzf --height=$height --header="Select containers to delete (Tab for multi-select, Enter to confirm):" --multi --ansi)

# Check if any containers were selected
if [ -z "$selected_containers" ]; then
    echo "No containers selected."
    exit 1
fi

# Loop through selected containers and stop & remove them
echo "$selected_containers" | awk '{print $1}' | while read container_id; do
    echo "Stopping container $container_id..."
    docker stop $container_id

    echo "Removing container $container_id..."
    docker rm $container_id || echo "Container $container_id already removed."
done

echo "Selected containers have been stopped and removed."