# Meeting Notes ASR service with FastAPI and Streamlit

A powerful Automatic Speech Recognition (ASR) service built with FastAPI, featuring semantic search capabilities, RAG (Retrieval-Augmented Generation), meeting notes analysis, and free-form LLM-enabled chat with Streamlit dashboards for UI all running as a service within one container on your machine, fully locally.

## 🌟 Features

- **Automatic Speech Recognition (ASR)**: Transcribe audio to text using whisper
- **Semantic Search**: Find relevant information in meeting transcripts and summaries
- **RAG (Retrieval-Augmented Generation)**: Get AI-generated answers based on your meeting content
- **Action Items Extraction**: Automatically identify and track action items from meetings
- **Background Processing**: Asynchronous task handling with Celery and Redis for batch processing meeting transcription and summarization
- **Meeting Notes Explorer**: Interactive UI for exploring and analyzing meeting notes
- **API service**: All functionality accessible to the UI is also exposed as an API to be used by other tools on the host machine as needed

## 📋 Requirements

- Python 3.11
- Docker
- Supports mp3 and m4a audio file types
- No GPU required (tested on Macbook Pro and Air with M2-M3 chips)

## 🚀 Quick Start

### Using Docker

1. Ensure [Ollama](https://ollama.com/download) is installed and running on your machine. 
2. Pull down necessary models.
    ```bash
   ollama pull qwen2.5:7b
   ollama pull all-minilm
   ```
3. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/asr-service-fast-api.git
   cd asr-service-fast-api
   ```
4. Create an empty .env file to set up Notion integration, if desired.
   - *Optional* Set up integration with [Notion](https://www.notion.com/) by defining variables in a .env file. Details at the end of the ReadMe. 
5. Build and run with Docker.
    ```bash
    docker build -t asr-service .
    docker run -p 8585:8000 -p 8586:8501 -p 8587:8502 -v --name asr-service-local --rm asr-service
    ```
    - *Optional* If you want to use the batch-processing functionality for mp3 files and save extracted meeting notes locally on your host machine use this docker run command instead
    ```bash
   docker run -p 8585:8000 -p 8586:8501 -p 8587:8502 -v /path/to/your/meeting-notes:/meeting-notes -v /path/to/your/meeting-recordings:/meeting-recordings --name asr-service-local --rm asr-service
    ```
6. Visit http://localhost:8586 and http://localhost:8587 to see meeting audio processor and explorer UI

## How-to

**Assumption**: You are recording your meetings as mp3 files and saving them to a directory. If you're not already doing so, it's easy to do this with [Piezo](https://rogueamoeba.com/piezo/). Can support any mp3 files with language in it. Multi-language support beyond english is present but not tested.

After setting up the service, you can use the two Streamlit UIs to process meeting recordings and explore meeting notes:

### 1. Meeting Audio Processor (http://localhost:8586)

This dashboard allows you to transcribe and summarize meeting recordings:

1. **Upload Individual Files**:
   - Navigate to the upload section
   - Upload an MP3 file of your meeting recording
   - Configure options (save to server, save to Notion)
   - Click "Process Audio" to start transcription and summarization
   - View real-time progress and results when complete

2. **Batch Processing**:
   - Place multiple MP3 files in your mounted `/meeting-recordings` directory
   - Navigate to the batch processing section
   - Configure options (save to server, save to Notion)
   - If using Notion integration, enter meeting date and participants
   - Click "Start Batch Processing"
   - The system will automatically chunk large files if needed
   - Monitor progress as files are processed sequentially

3. **View Results**:
   - After processing, transcripts and summaries are saved to the mounted `/meeting-notes` directory
   - Results are displayed directly in the UI when processing completes

### 2. Meeting Notes Explorer (http://localhost:8587)

This dashboard provides multiple ways to interact with your processed meeting notes:

1. **Meeting Viewer**:
   - Select a meeting from the dropdown menu
   - View the meeting summary in the "Meeting Summary" tab
   - View the full transcript in the "Full Transcript" tab
   - Download either the summary or transcript as needed

2. **Search Interface**:
   - Enter keywords or phrases in the search box
   - Choose search type:
     - **Keyword Search**: Finds exact matches of your search terms
     - **Semantic Search**: Finds content with similar meaning using embeddings
     - **Smart Search (AI-powered)**: Uses RAG to understand your query and provide intelligent answers
   - Filter by date range to narrow down results
   - Select specific meetings to search in
   - View highlighted matches in both summaries and transcripts
   - For semantic and RAG searches, see relevance scores for each match

3. **Chat Interface**:
   - Have a conversational interaction with your meeting content
   - Ask questions in natural language about any of your meetings
   - Filter which meetings to include in the context (by date or specific selection)
   - View the AI's responses based on your meeting content
   - See the specific context used to generate answers
   - Maintain conversation history for contextual follow-up questions

4. **Analytics**:
   - View statistics about your meeting notes collection
   - See trends in meeting frequency, duration, and participation
   - Explore word clouds of common terms used across meetings
   - Identify frequent topics and action items

5. **Settings**:
   - Manage semantic search index
   - Rebuild search indices as needed
   - View index statistics

### 3. API Access (http://localhost:8585)

All functionality is also available via API endpoints:

1. **API Documentation**:
   - Visit http://localhost:8585/docs for interactive Swagger documentation
   - Or http://localhost:8585/redoc for ReDoc documentation

2. **Key Endpoints**:
   - Upload and process audio files
   - Search meeting transcripts and summaries
   - Retrieve meeting notes
   - Extract action items
   - Perform RAG-based queries

This workflow allows you to efficiently process meeting recordings, explore the resulting notes through various interfaces, and leverage AI capabilities to extract maximum value from your meeting content.

## Optional: Notion API Integration
Set up integration with [Notion](https://www.notion.com/) by creating a `.env` file in the project root directory:

    NOTION_TOKEN=your_notion_integration_token
    NOTION_DATABASE_ID=your_notion_database_id

To set up Notion integration:
1. Create a Notion integration at https://www.notion.so/my-integrations
2. Copy the "Internal Integration Token" to use as your `NOTION_TOKEN`
3. Create a database in Notion to store your meeting notes
4. Share your database with the integration you created
5. Get your database ID from the URL of your database page:
   - The URL format is: `https://www.notion.so/{workspace}/{database_id}?v={view_id}`
   - Copy the part between the workspace name and the question mark as your `NOTION_DATABASE_ID`

When enabled, meeting transcripts and summaries will be automatically saved to your Notion database.

## Citation

If you use this project in your research or work, please cite it using the following BibTeX entries:

```bibtex
@misc{awesome-asr-service,
  author = {Bhattacharya, Kiran},
  title = {Meeting Notes ASR Service and Dataset},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/SKA-LAB/awesome-asr-service}},
}