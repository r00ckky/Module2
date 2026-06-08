LOCAL_FOLDER="/home/chaitanya-kohli/AgenticAI/Module2"
REMOTE_DEST="/DATA/Module2"
SSH_USER="teaching"

REMOTE_MACHINES=(
    "172.18.40.101"
    "172.18.40.102"
    "172.18.40.103"
    "172.18.40.104"
    "172.18.40.105"
    "172.18.40.106"
    "172.18.40.107"
    "172.18.40.108"
    "172.18.40.109"
    "172.18.40.110"
    "172.18.40.111"
    "172.18.40.112"
    "172.18.40.113"
    "172.18.40.114"
    "172.18.40.115"
    "172.18.40.116"
    "172.18.40.117"
    "172.18.40.118"
    "172.18.40.119"
    "172.18.40.120"
    "172.18.40.121"
    "172.18.40.122"
    "172.18.40.123"
    "172.18.40.124"
    "172.18.40.125"
    "172.18.40.126"
    "172.18.40.127"
    "172.18.40.128"
    "172.18.40.129"
    "172.18.40.130"
    "172.18.40.131"
    "172.18.40.132"
    "172.18.40.133"
    "172.18.40.134"
    "172.18.40.135"
    "172.18.40.136"
    "172.18.40.137"
    "172.18.40.138"
    "172.18.40.139"
    "172.18.40.140"
)

PASSWORDS=("dslab123" "ds123")

if ! command -v sshpass &> /dev/null; then
    echo "❌ Error: 'sshpass' is not installed locally. Run: sudo apt install sshpass"
    exit 1
fi

echo "🚀 Starting deployment and environment check..."
echo "------------------------------------------------"

for MACHINE in "${REMOTE_MACHINES[@]}"; do
    echo "🔄 Connecting to $MACHINE..."
    SUCCESS=false
    WORKING_PASS=""

    # 1. FIND WORKING PASSWORD & COPY FOLDER
    for PASS in "${PASSWORDS[@]}"; do
        sshpass -p "$PASS" scp -r -o StrictHostKeyChecking=no "$LOCAL_FOLDER" "${SSH_USER}@${MACHINE}:${REMOTE_DEST}" &> /dev/null
        
        if [ $? -eq 0 ]; then
            echo "  ✅ Success: Folder copied."
            SUCCESS=true
            WORKING_PASS="$PASS"
            break 
        fi
    done

    # 2. CHECK FOR OLLAMA (Only if connection succeeded)
    if [ "$SUCCESS" = true ]; then
        # Run a remote command to check if 'ollama' exists in the PATH
        sshpass -p "$WORKING_PASS" ssh -o StrictHostKeyChecking=no "${SSH_USER}@${MACHINE}" "command -v ollama" &> /dev/null
        
        if [ $? -eq 0 ]; then
            echo "  🦙 Ollama Status: [ INSTALLED ]"
        else
            echo "  ⚠️  Ollama Status: [ NOT INSTALLED ]"
        fi
    else
        echo "  ❌ Failure: Could not connect with provided passwords."
    fi
    echo "------------------------------------------------"
done