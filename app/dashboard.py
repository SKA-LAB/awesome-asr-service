import streamlit as st
import requests
import time
import os
import subprocess
import glob
from io import BytesIO
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Meeting Transcription & Summary",
    page_icon="🎙️",
    layout="wide"
)

# Define API endpoint
API_URL = os.environ.get("API_URL", "http://localhost:8000")
# Define the directory where MP3 files are stored for batch processing
BATCH_DIRECTORY = os.environ.get("BATCH_DIRECTORY", "/path/to/mp3/files")
# Path to the chunking script
CHUNK_SCRIPT = os.environ.get("CHUNK_SCRIPT", "/path/to/chunk_mp3.sh")

def process_single_file(file_path, save_to_file=False, save_to_notion=False, meeting_date=None, participants_list=None):
    """Process a single MP3 file and return the results"""
    try:
        # Read the file
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Get filename
        filename = os.path.basename(file_path)
        
        # Prepare the file for upload
        files = {"file": (filename, file_content, "audio/mpeg")}
        
        # Prepare parameters
        params = {
            "save_to_file": save_to_file,
            "save_to_notion": save_to_notion
        }
        
        # Add Notion-specific parameters if needed
        if save_to_notion and meeting_date:
            params["meeting_date"] = meeting_date.strftime("%Y-%m-%d")
            if participants_list and participants_list[0]:
                params["participants"] = participants_list
        
        # Make API request
        response = requests.post(
            f"{API_URL}/transcribe-and-summarize/",
            files=files,
            params=params,
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Error: {response.status_code} - {response.text}"}
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

def batch_process_directory():
    """Process all MP3 files in the batch directory"""
    st.subheader("Batch Processing")
    
    # Settings for batch processing
    col1, col2 = st.columns(2)
    with col1:
        save_to_file = st.toggle("Save to server", value=True, 
                                help="When enabled, the transcript and summary will be saved as markdown files on the server")
    with col2:
        save_to_notion = st.toggle("Save to Notion", value=False,
                                  help="When enabled, the transcript and summary will be saved to your Notion workspace")
    
    # If Notion is selected, show additional options
    if save_to_notion:
        st.subheader("Notion Details")
        meeting_date = st.date_input("Meeting Date", datetime.now())
        participants = st.text_input("Participants (comma separated)", "")
        
        # Convert participants string to list
        participants_list = [p.strip() for p in participants.split(",")] if participants else []
    else:
        meeting_date = None
        participants_list = None
    
    # Check if directory exists
    if not os.path.exists(BATCH_DIRECTORY):
        st.error(f"Batch directory not found: {BATCH_DIRECTORY}")
        return
    
    # Count MP3 files in directory
    mp3_files = glob.glob(os.path.join(BATCH_DIRECTORY, "*.mp3"))
    if not mp3_files:
        st.warning(f"No MP3 files found in {BATCH_DIRECTORY}")
        return
    
    st.info(f"Found {len(mp3_files)} MP3 files in {BATCH_DIRECTORY}")
    
    # Start batch processing button
    if st.button("Start Batch Processing"):
        # Run the chunking script first
        with st.spinner("Running MP3 chunking script..."):
            try:
                # Create a placeholder for script output
                script_output = st.empty()
                script_output.text("Starting chunking script...")
                
                # Run the chunking script
                process = subprocess.Popen(
                    [CHUNK_SCRIPT, BATCH_DIRECTORY],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # Stream the output
                output_lines = []
                for line in iter(process.stdout.readline, ''):
                    if not line:
                        break
                    output_lines.append(line.strip())
                    script_output.text('\n'.join(output_lines[-10:]))  # Show last 10 lines
                
                # Wait for the process to complete
                return_code = process.wait()
                
                if return_code != 0:
                    st.error(f"Chunking script failed with return code {return_code}")
                    return
                else:
                    st.success("Chunking script completed successfully")
            
            except Exception as e:
                st.error(f"Error running chunking script: {str(e)}")
                return
        
        # Refresh the list of MP3 files after chunking
        mp3_files = glob.glob(os.path.join(BATCH_DIRECTORY, "*.mp3"))
        st.info(f"Processing {len(mp3_files)} MP3 files...")
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process each file
        results = []
        for i, file_path in enumerate(mp3_files):
            filename = os.path.basename(file_path)
            status_text.text(f"Processing file {i+1}/{len(mp3_files)}: {filename}")
            
            # Process the file
            result = process_single_file(
                file_path, 
                save_to_file=save_to_file, 
                save_to_notion=save_to_notion,
                meeting_date=meeting_date,
                participants_list=participants_list
            )
            
            # Store result
            result["filename"] = filename
            results.append(result)
            
            # Update progress
            progress_bar.progress((i + 1) / len(mp3_files))
        
        # Delete original MP3 files
        status_text.text("Cleaning up original MP3 files...")
        for file_path in mp3_files:
            try:
                os.remove(file_path)
            except Exception as e:
                st.warning(f"Could not delete {file_path}: {str(e)}")
        
        status_text.text("Batch processing completed!")
        
        # Display results
        st.subheader("Processing Results")
        for result in results:
            with st.expander(f"Result for {result['filename']}"):
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.success("Processing successful")
                    if save_to_file and result.get("saved_to_file"):
                        st.info(f"Saved to: {result['saved_to_file']}")
                    if save_to_notion and result.get("saved_to_notion"):
                        st.info(f"Saved to Notion: {result['saved_to_notion']}")

def main():
    st.title("🎙️ Meeting Transcription & Summary")
    
    # Create tabs for single file and batch processing
    tab1, tab2 = st.tabs(["Single File Processing", "Batch Directory Processing"])
    
    with tab1:
        st.write("Upload an MP3 recording of your meeting to get a transcript and summary.")
        
        # File uploader
        uploaded_file = st.file_uploader("Upload MP3 file", type=["mp3"])
        
        if uploaded_file is not None:
            # Display file info
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.write("File Details:")
            for key, value in file_details.items():
                st.write(f"- {key}: {value}")
            
            # Add toggle for saving options
            col1, col2 = st.columns(2)
            with col1:
                save_to_file = st.toggle("Save to server", value=False, 
                                        help="When enabled, the transcript and summary will be saved as a markdown file on the server")
            with col2:
                save_to_notion = st.toggle("Save to Notion", value=False,
                                          help="When enabled, the transcript and summary will be saved to your Notion workspace")
            
            # If Notion is selected, show additional options
            if save_to_notion:
                st.subheader("Notion Details")
                meeting_date = st.date_input("Meeting Date", datetime.now())
                participants = st.text_input("Participants (comma separated)", "")
                
                # Convert participants string to list
                participants_list = [p.strip() for p in participants.split(",")] if participants else []
            else:
                meeting_date = None
                participants_list = None
            
            # Process button
            if st.button("Transcribe and Summarize"):
                with st.spinner("Processing your audio file... This may take a few minutes."):
                    try:
                        # Prepare the file for upload
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/mpeg")}
                        
                        # Prepare parameters
                        params = {
                            "save_to_file": save_to_file,
                            "save_to_notion": save_to_notion
                        }
                        
                        # Add Notion-specific parameters if needed
                        if save_to_notion:
                            params["meeting_date"] = meeting_date.strftime("%Y-%m-%d")
                            if participants_list and participants_list[0]:  # Check if there's at least one non-empty participant
                                params["participants"] = participants_list
                        
                        # Make API request with save_to_file parameter
                        response = requests.post(
                            f"{API_URL}/transcribe-and-summarize/",
                            files=files,
                            params=params,
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            
                            # Create tabs for transcript and summary
                            tab1, tab2 = st.tabs(["Meeting Summary", "Full Transcript"])
                            
                            with tab1:
                                st.markdown(result["summary_markdown"])
                                
                                # Add download button for summary
                                summary_bytes = BytesIO(result["summary_markdown"].encode())
                                st.download_button(
                                    label="Download Summary",
                                    data=summary_bytes,
                                    file_name=f"{uploaded_file.name.split('.')[0]}_summary.md",
                                    mime="text/markdown"
                                )
                            
                            with tab2:
                                st.subheader("Full Transcript")
                                st.text_area("", value=result["transcription"], height=400)
                                
                                # Add download button for transcript
                                transcript_bytes = BytesIO(result["transcription"].encode())
                                st.download_button(
                                    label="Download Transcript",
                                    data=transcript_bytes,
                                    file_name=f"{uploaded_file.name.split('.')[0]}_transcript.txt",
                                    mime="text/plain"
                                )
                            
                            # Display confirmation if file was saved on server
                            if save_to_file and result.get("saved_to_file"):
                                st.success(f"Transcript and summary saved to: {result['saved_to_file']}")
                        else:
                            st.error(f"Error: {response.status_code} - {response.text}")
                    
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")
    
    with tab2:
        batch_process_directory()
    
    # Add information section
    with st.expander("ℹ️ About this tool"):
        st.write("""
        This tool uses automatic speech recognition (ASR) to transcribe your meeting recordings and 
        AI to generate a concise summary with action items. The processing happens on our secure servers.
        
        **Supported file format:** MP3 audio files
        
        **Privacy note:** Your audio files are processed and then immediately deleted from our servers.
        
        **Save to server option:** When enabled, a markdown file containing both the transcript and summary 
        will be saved on the server for future reference.
        
        **Batch processing:** The batch processing tab allows you to process all MP3 files in a specified directory.
        It first runs a chunking script to prepare the files, then processes each file individually, and finally
        deletes the original MP3 files to save space.
        """)

if __name__ == "__main__":
    main()