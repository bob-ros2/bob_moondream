#!/bin/bash

TYPE=${1:-"visual_query"}
NAMESPACE=${2:-"/vision"}

# The topic to publish prompts to
PROMPT_TOPIC=${NAMESPACE}/${TYPE}_prompt
# The topic to listen for results on
RESULT_TOPIC=${NAMESPACE}/${TYPE}_result

[ $# -lt 1 ] && echo "Usage: $(basename $0) <visual_query|object_detection|pointing> [<namespace>]" && exit 1

cleanup() {
    echo -e "\n\nShutting down listener and exiting..."
    if kill -0 "$LISTENER_PID" 2>/dev/null; then
        kill "$LISTENER_PID"
    fi
    exit 0
}

trap cleanup SIGINT

# Start listening to the result topic in the background
echo "--- Listening for results on $RESULT_TOPIC ---"
ros2 topic echo -f "$RESULT_TOPIC" std_msgs/msg/String | sed 's/^data: //g' &
LISTENER_PID=$!

echo "--- Enter your prompt below (Press Ctrl+C to exit) ---"
while true; do
    # Prompt the user for input
    read -p "> " prompt

    if [[ -z "$prompt" ]]; then
        continue
    fi

    ros2 topic pub --keep-alive 0.5 --once "$PROMPT_TOPIC" std_msgs/msg/String "data: '$prompt'"
done