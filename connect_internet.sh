#!/bin/bash

# Execute curl and direct output to wget-logs file
curl --location-trusted -u 21d100003:26df5f33a3018cf27eadb2c14e80fdd9 "https://internet-sso.iitb.ac.in/login.php" > ./logs/curl-logs

# Check the exit status of curl command
if [ $? -eq 0 ]; then
    echo 'Logged in!'
    # Optionally, you can also append this message to the wget-logs file
    echo 'Logged in!' >> ./logs/curl-logs
else
    echo 'Something is wrong or already logged in!'
    # Optionally, append error message to the wget-logs file
    echo 'Something is wrong or already logged in!' >> ./logs/curl-logs
fi
