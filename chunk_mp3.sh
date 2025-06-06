#!/bin/bash
# File: /Users/kbhattacha/Documents/asr-service-fast-api/chunk_mp3.sh

# Set default values
MAX_DURATION=3600  # Maximum duration in seconds (30 minutes)
OVERLAP=5          # Overlap between chunks in seconds
OUTPUT_DIR=""      # By default, output to the same directory as input

# Function to display usage information
usage() {
    echo "Usage: $0 [OPTIONS] INPUT_DIRECTORY"
    echo
    echo "Options:"
    echo "  -m, --max-duration SECONDS   Maximum duration of each chunk in seconds (default: 1800)"
    echo "  -o, --overlap SECONDS        Overlap between chunks in seconds (default: 5)"
    echo "  -d, --output-dir DIRECTORY   Output directory (default: same as input)"
    echo "  -h, --help                   Display this help message"
    echo
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--max-duration)
            MAX_DURATION="$2"
            shift 2
            ;;
        -o|--overlap)
            OVERLAP="$2"
            shift 2
            ;;
        -d|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            INPUT_DIR="$1"
            shift
            ;;
    esac
done

# Check if input directory is provided
if [ -z "$INPUT_DIR" ]; then
    echo "Error: Input directory not specified"
    usage
fi

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist"
    exit 1
fi

# If output directory is not specified, use input directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$INPUT_DIR"
fi

# Create output directory if it doesn't exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create output directory '$OUTPUT_DIR'"
        exit 1
    fi
    echo "Created output directory: $OUTPUT_DIR"
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "Error: ffmpeg is not installed. Please install ffmpeg to use this script."
    exit 1
fi

# Check if ffprobe is installed
if ! command -v ffprobe &> /dev/null; then
    echo "Error: ffprobe is not installed. Please install ffprobe to use this script."
    exit 1
fi

# Function to format time in HH:MM:SS
format_time() {
    local seconds=$1
    printf "%02d:%02d:%02d" $((seconds/3600)) $((seconds%3600/60)) $((seconds%60))
}

# First, convert any M4A files to MP3
echo "Scanning for M4A files in $INPUT_DIR..."
find "$INPUT_DIR" -type f -name "*.m4a" | while read -r file; do
    filename=$(basename "$file")
    base_name="${filename%.*}"
    mp3_file="$INPUT_DIR/${base_name}.mp3"
    
    echo "Converting: $filename to MP3"
    
    # Convert M4A to MP3
    ffmpeg -v warning -i "$file" -acodec libmp3lame -q:a 2 "$mp3_file"
    
    if [ $? -ne 0 ]; then
        echo "  Error: Failed to convert $filename to MP3"
    else
        echo "  Successfully converted to: $(basename "$mp3_file")"
        
        # Delete the original M4A file
        rm "$file"
        if [ $? -ne 0 ]; then
            echo "  Warning: Failed to delete original M4A file: $filename"
        else
            echo "  Original M4A file deleted successfully"
        fi
    fi
    
    echo
done

# Process each MP3 file in the input directory
echo "Scanning for MP3 files in $INPUT_DIR..."
find "$INPUT_DIR" -type f -name "*.mp3" | while read -r file; do
    filename=$(basename "$file")
    echo "Processing: $filename"
    
    # Get duration of the file in seconds
    duration=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file")
    duration_seconds=$(printf "%.0f" "$duration")
    
    echo "  Duration: $(format_time $duration_seconds) ($duration_seconds seconds)"
    
    # If file is shorter than max duration, skip chunking
    if [ "$duration_seconds" -le "$MAX_DURATION" ]; then
        echo "  File is shorter than maximum duration ($MAX_DURATION seconds), skipping chunking"
        continue
    fi
    
    # Calculate number of chunks
    num_chunks=$(( (duration_seconds + MAX_DURATION - 1) / MAX_DURATION ))
    echo "  Splitting into $num_chunks chunks of maximum $(format_time $MAX_DURATION) each"
    
    # Create a temporary directory for this file
    temp_dir=$(mktemp -d)
    base_name="${filename%.*}"
    
    # Flag to track if all chunks were created successfully
    all_chunks_successful=true
    
    # Split the file into chunks
    for (( i=0; i<num_chunks; i++ )); do
        start_time=$((i * MAX_DURATION - (i > 0 ? OVERLAP : 0)))
        if [ $start_time -lt 0 ]; then
            start_time=0
        fi
        
        # For the last chunk, use the full duration
        if [ $((i + 1)) -eq $num_chunks ]; then
            chunk_file="$OUTPUT_DIR/${base_name}_chunk$((i+1))of$num_chunks.mp3"
            echo "  Creating chunk $((i+1))/$num_chunks: $(format_time $start_time) to end"
            ffmpeg -v warning -i "$file" -ss $(format_time $start_time) -acodec copy "$chunk_file"
        else
            # For other chunks, use the specified duration
            chunk_duration=$((MAX_DURATION + OVERLAP))
            chunk_file="$OUTPUT_DIR/${base_name}_chunk$((i+1))of$num_chunks.mp3"
            echo "  Creating chunk $((i+1))/$num_chunks: $(format_time $start_time) to $(format_time $((start_time + MAX_DURATION)))"
            ffmpeg -v warning -i "$file" -ss $(format_time $start_time) -t $(format_time $chunk_duration) -acodec copy "$chunk_file"
        fi
        
        if [ $? -ne 0 ]; then
            echo "  Error: Failed to create chunk $((i+1))/$num_chunks"
            all_chunks_successful=false
        else
            echo "  Successfully created: $(basename "$chunk_file")"
        fi
    done
    
    # Remove the temporary directory
    rm -rf "$temp_dir"
    
    # Delete the original file if all chunks were created successfully
    if [ "$all_chunks_successful" = true ]; then
        echo "  All chunks created successfully. Deleting original file: $filename"
        rm "$file"
        if [ $? -ne 0 ]; then
            echo "  Warning: Failed to delete original file: $filename"
        else
            echo "  Original file deleted successfully"
        fi
    else
        echo "  Warning: Some chunks failed to create. Original file will not be deleted."
    fi
    
    echo "  Finished processing $filename"
    echo
done

echo "All audio files processed successfully!"
exit 0