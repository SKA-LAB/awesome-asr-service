import streamlit as st
import requests
import time
import os
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Meeting Transcription & Summary",
    page_icon="🎙️",
    layout="wide"
)

# Define API endpoint
API_URL = os.environ.get("API_URL", "http://localhost:8000")

def main():
    st.title("🎙️ Meeting Transcription & Summary")
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
        
        # Process button
        if st.button("Transcribe and Summarize"):
            with st.spinner("Processing your audio file... This may take a few minutes."):
                try:
                    # Prepare the file for upload
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "audio/mpeg")}
                    
                    # Make API request
                    response = requests.post(
                        f"{API_URL}/transcribe-and-summarize/",
                        files=files
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
                    else:
                        st.error(f"Error: {response.status_code} - {response.text}")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
    
    # Add information section
    with st.expander("ℹ️ About this tool"):
        st.write("""
        This tool uses automatic speech recognition (ASR) to transcribe your meeting recordings and 
        AI to generate a concise summary with action items. The processing happens on our secure servers.
        
        **Supported file format:** MP3 audio files
        
        **Privacy note:** Your audio files are processed and then immediately deleted from our servers.
        """)

if __name__ == "__main__":
    main()