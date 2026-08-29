#!/bin/bash

# RUNNING COMMANDS: nohup ./rclone_download.sh >> rclone.log 2>&1 & 

# GLOBAL VARS
SRC_DIR="thesis_data:DataThesis/zipped"
DEST_DIR="$HOME/data/zipped"
FILTER_FILE="$HOME/rclone_filters.txt"
LOG_FILE="$HOME/download_$(date +%Y%m%d_%H%M%S).log"

# DECLARE FILTERS
cat <<EOF > "$FILTER_FILE"
+ /MerlMix/merlmix.zip
+ /PolarPS/obj_ax.zip
- *
EOF

printf '=%.0s' {1..50}
echo
echo "DATE: $(date)"
echo "Initializing rclone download..."
echo "Detailed logs will be written to: $LOG_FILE"

# rclone copy
rclone copy "$SRC_DIR" "$DEST_DIR" \
	--filter-from "$FILTER_FILE" \
	--log-file "$LOG_FILE" \
	--log-level INFO \
	--stats 30s --stats-one-line-date \
	--transfers 2 \
	--checkers 4 \
	--low-level-retries 20 \
	--retries 10 \
	--timeout 5m \
	--contimeout 1m \
	--checksum

# VERIFICATION
RCLONE_EXIT=$?
if [ $RCLONE_EXIT -eq 0 ]; then
	echo "Download and checksum verification completed!"
	exit 0
else
	echo "Download failed. rclone exited with code $EXIT_CODE" >&2
	echo "Please review the log file for details: $LOG_FILE" >&2
	exit $RCLONE_EXIT
fi
