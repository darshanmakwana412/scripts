#!/bin/bash

scripts_dir=~/scripts

echo "Fetching bash scripts in $scripts_dir..."

# Find all .sh files in the specified directory, get their basename
scripts=$(find "$scripts_dir" -type f -name "*.sh" ! -name "entrypoint.sh" ! -name "main.sh" -exec basename {} \;)

# Calculate the number of scripts, but limit the maximum height
script_count=$(echo "$scripts" | wc -l)
max_height=10
height=$((script_count+4)) # Add a little padding for fzf's UI elements
if [ $height -gt $max_height ]; then
    height=$max_height
fi

# Use fzf to let the user select a script by name
selected_script_name=$(echo "$scripts" | fzf --height=$height --header="Select a script to run:")

# Find the full path of the selected script
selected_script_path=$(find "$scripts_dir" -type f -name "$selected_script_name")

# Check if a script was selected
if [ -z "$selected_script_path" ]; then
    echo "No script selected."
    exit 1
fi

echo "Running script: $selected_script_path"
bash "$selected_script_path"