import os
import re
import streamlit as st
import faiss
import glob
from datetime import datetime
import pandas as pd
from pathlib import Path
import concurrent.futures
from functools import partial
import indexing_utils
import time
from semantic_search import get_search_engine
from rag_search import get_rag_search_engine
import torch

torch.classes.__path__ = [] # add this line to manually set it to empty.

# Initialize session state for search type
if 'search_type' not in st.session_state:
    st.session_state.search_type = "keyword"

# Define constants for semantic search
SEMANTIC_SEARCH_ENABLED = True
RAG_SEARCH_ENABLED = True

# Set page configuration
st.set_page_config(
    page_title="Meeting Notes Explorer",
    page_icon="📝",
    layout="wide"
)

# Define constants
MEETING_NOTES_DIR = os.environ.get("OUTPUT_DIR", "/meeting-notes")

def load_meeting_files():
    """Load all meeting markdown files from the meeting notes directory."""
    files = glob.glob(f"{MEETING_NOTES_DIR}/*.md")
    meeting_files = []
    
    for file_path in files:
        file_name = os.path.basename(file_path)
        # Extract date from filename (assuming format: name_YYYYMMDD-HHMMSS.md)
        date_match = re.search(r'_(\d{8}-\d{6})\.md$', file_name)
        
        if date_match:
            date_str = date_match.group(1)
            try:
                date = datetime.strptime(date_str, "%Y%m%d-%H%M%S")
                display_name = file_name.replace('.md', '').replace('_', ' ').replace(date_str, '')
                display_name = f"{display_name.strip()} ({date.strftime('%Y-%m-%d %H:%M')})"
            except ValueError:
                display_name = file_name
                date = datetime.fromtimestamp(os.path.getmtime(file_path))
        else:
            display_name = file_name
            date = datetime.fromtimestamp(os.path.getmtime(file_path))
            
        meeting_files.append({
            "path": file_path,
            "name": file_name,
            "display_name": display_name,
            "date": date
        })
    
    # Sort by date, newest first
    meeting_files.sort(key=lambda x: x["date"], reverse=True)
    return meeting_files

def parse_markdown_file(file_path):
    """Parse a markdown file into sections."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Extract title
    title_match = re.search(r'^# (.+)$', content, re.MULTILINE)
    title = title_match.group(1) if title_match else "Untitled Meeting"
    
    # Extract transcript
    transcript_match = re.search(r'## Full Transcript\s+```\s+(.*?)\s+```', content, re.DOTALL)
    transcript = transcript_match.group(1) if transcript_match else ""
    
    # Extract summary
    summary_match = re.search(r'## Summary\s+(.*?)(?=\Z|\n## )', content, re.DOTALL)
    summary = summary_match.group(1) if summary_match else ""
    
    return {
        "title": title,
        "transcript": transcript,
        "summary": summary
    }

def search_in_file(file_info, query_lower, search_in_transcripts=True, search_in_summaries=True):
    """Search for a query in a single file."""
    file_path = file_info["path"]
    content = parse_markdown_file(file_path)
    
    # Search in summary
    summary_matches = []
    if search_in_summaries and content["summary"]:
        summary_lines = content["summary"].split('\n')
        for i, line in enumerate(summary_lines):
            if query_lower in line.lower():
                start = max(0, i - 1)
                end = min(len(summary_lines), i + 2)
                context = '\n'.join(summary_lines[start:end])
                summary_matches.append({
                    "context": context,
                    "type": "summary"
                })
    
    # Search in transcript
    transcript_matches = []
    if search_in_transcripts and content["transcript"]:
        transcript_lines = content["transcript"].split('\n')
        for i, line in enumerate(transcript_lines):
            if query_lower in line.lower():
                start = max(0, i - 2)
                end = min(len(transcript_lines), i + 3)
                context = '\n'.join(transcript_lines[start:end])
                transcript_matches.append({
                    "context": context,
                    "type": "transcript"
                })
    
    # If we found matches, return result
    if summary_matches or transcript_matches:
        return {
            "file_info": file_info,
            "summary_matches": summary_matches,
            "transcript_matches": transcript_matches,
            "match_count": len(summary_matches) + len(transcript_matches)
        }
    return None


def search_in_files(files, query, search_in_transcripts=True, search_in_summaries=True):
    """Search for a query in the specified files using keyword or semantic search."""
    if not query.strip():
        return []
    
    # Determine search type from session state
    search_type = st.session_state.get('search_type', 'keyword')
    
    if search_type == "semantic" and SEMANTIC_SEARCH_ENABLED:
        # Perform semantic search
        try:
            # First, ensure all files are indexed
            with st.spinner("Indexing files for semantic search..."):
                indexing_utils.index_meeting_files(files)
            
            # Then perform the search
            with st.spinner("Performing semantic search..."):
                results = indexing_utils.perform_semantic_search(
                    query,
                    files,
                    search_in_summaries=search_in_summaries,
                    search_in_transcripts=search_in_transcripts
                )
            return results
        except Exception as e:
            st.error(f"Semantic search failed: {str(e)}")
            # Fall back to keyword search
            st.warning("Falling back to keyword search")
            search_type = "keyword"
    
    # Keyword search (original implementation)
    query_lower = query.lower()
    results = []
    
    # Create a partial function with fixed parameters
    search_func = partial(
        search_in_file, 
        query_lower=query_lower, 
        search_in_transcripts=search_in_transcripts, 
        search_in_summaries=search_in_summaries
    )
    
    # Use ThreadPoolExecutor to parallelize the search
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Submit all search tasks and collect futures
        future_to_file = {executor.submit(search_func, file_info): file_info for file_info in files}
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            result = future.result()
            if result:  # Only add non-None results
                results.append(result)
    
    # Sort results by number of matches
    results.sort(key=lambda x: x["match_count"], reverse=True)
    return results


def ensure_files_indexed(files):
    """Ensure all files are indexed for semantic search."""
    if st.session_state.search_type == "semantic":
        try:
            indexing_utils.index_meeting_files(files)
            return True
        except Exception as e:
            st.error(f"Failed to index files: {str(e)}")
            return False
    return True

def display_meeting_viewer():
    """Display the meeting notes viewer interface."""
    st.header("📋 Meeting Notes Viewer")
    
    # Load meeting files
    meeting_files = load_meeting_files()
    
    if not meeting_files:
        st.warning("No meeting notes found. Please check the meeting notes directory.")
        return
    
    # Create a dropdown to select meeting
    meeting_options = [file["display_name"] for file in meeting_files]
    meeting_options.reverse()
    selected_meeting = st.selectbox("Select a meeting", meeting_options)
    
    # Find the selected meeting file
    selected_file = next((file for file in meeting_files if file["display_name"] == selected_meeting), None)
    
    if selected_file:
        # Parse and display the meeting content
        content = parse_markdown_file(selected_file["path"])
        
        # Display meeting info
        st.subheader(content["title"])
        st.caption(f"File: {selected_file['name']}")
        
        # Create tabs for summary and transcript
        summary_tab, transcript_tab = st.tabs(["Meeting Summary", "Full Transcript"])
        
        with summary_tab:
            st.markdown(content["summary"])
            
            # Add download button for summary
            st.download_button(
                label="Download Summary",
                data=content["summary"],
                file_name=f"{selected_file['name'].replace('.md', '')}_summary.md",
                mime="text/markdown"
            )
        
        with transcript_tab:
            st.text_area("", value=content["transcript"], height=400)
            
            # Add download button for transcript
            st.download_button(
                label="Download Transcript",
                data=content["transcript"],
                file_name=f"{selected_file['name'].replace('.md', '')}_transcript.txt",
                mime="text/plain"
            )

def display_search_interface():
    """Display the search interface."""
    st.header("🔍 Search Meeting Notes")
    
    # Load meeting files
    meeting_files = load_meeting_files()
    
    if not meeting_files:
        st.warning("No meeting notes found. Please check the meeting notes directory.")
        return
    
    # Create search filters
    col1, col2 = st.columns([3, 1])
    
    with col1:
        search_query = st.text_input("Search for keywords or phrases", key="search_query")
    
    with col2:
        st.write("Search in:")
        search_in_summaries = st.checkbox("Summaries", value=True)
        search_in_transcripts = st.checkbox("Transcripts", value=True)
    
    # Add search type selector
    search_type = st.radio(
        "Search Type",
        ["Keyword Search", "Semantic Search", "Smart Search (AI-powered)"],
        horizontal=True,
        index=0 if st.session_state.search_type == "keyword" else 
              1 if st.session_state.search_type == "semantic" else 2
    )
    if search_type == "Keyword Search":
        st.session_state.search_type = "keyword"
    elif search_type == "Semantic Search":
        st.session_state.search_type = "semantic"
    else:
        st.session_state.search_type = "rag"
    
    # Add help text for search types
    if st.session_state.search_type == "keyword":
        st.caption("Keyword search finds exact matches of your search terms.")
    elif st.session_state.search_type == "semantic":
        st.caption("Semantic search finds content with similar meaning, even if the exact words are different.")
    else:  # RAG search
        st.caption("RAG search uses AI to understand your query and provide intelligent answers based on meeting content.")
    
    # Add date range filter
    st.subheader("Filter by Date")
    
    # Get min and max dates from files
    dates = [file["date"] for file in meeting_files]
    min_date = min(dates).date() if dates else datetime.now().date()
    max_date = max(dates).date() if dates else datetime.now().date()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("From", min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)
    
    # Filter files by date
    filtered_files = [
        file for file in meeting_files 
        if start_date <= file["date"].date() <= end_date
    ]
    
    # Allow selecting specific meetings to search in
    st.subheader("Select Meetings to Search")
    
    # Create a dataframe for better display
    df = pd.DataFrame([
        {"Meeting": file["display_name"], "Date": file["date"].strftime("%Y-%m-%d %H:%M")}
        for file in filtered_files
    ])
    
    # Display as a table with selection
    selection = st.multiselect(
        "Select specific meetings (leave empty to search all)",
        options=df["Meeting"].tolist(),
        default=[]
    )
    
    # Filter files based on selection
    if selection:
        search_files = [file for file in filtered_files if file["display_name"] in selection]
    else:
        search_files = filtered_files
    
    st.caption(f"Searching in {len(search_files)} meeting notes")
    
    # Add index management for semantic search
    if st.session_state.search_type in ["semantic", "rag"]:
        with st.expander("Semantic Search Index Management"):
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Rebuild Search Index"):
                    with st.spinner("Rebuilding search index..."):
                        try:
                            search_engine = get_search_engine()
                            search_engine.reindex_all(meeting_files)
                            st.success("Search index rebuilt successfully!")
                        except Exception as e:
                            st.error(f"Failed to rebuild index: {str(e)}")
            
            with col2:
                index_stats = get_search_engine()
                st.metric("Indexed Summaries", index_stats.summary_index.ntotal if index_stats.summary_index else 0)
                st.metric("Indexed Transcript Chunks", index_stats.transcript_index.ntotal if index_stats.transcript_index else 0)
    
    # Search button
    if st.button("Search", key="search_button") or search_query:
        if not search_query:
            st.warning("Please enter a search query")
        elif not (search_in_summaries or search_in_transcripts):
            st.warning("Please select at least one area to search in (Summaries or Transcripts)")
        else:
            # Show search type
            search_type_display = "Smart Search (AI-powered)" if st.session_state.search_type == "rag" else \
                                 "Semantic Search" if st.session_state.search_type == "semantic" else "Keyword Search"
            st.info(f"Performing {search_type_display} for '{search_query}'")
            
            # Perform search
            start_time = time.time()
            if st.session_state.search_type == "rag":
                # Perform RAG search
                with st.spinner("AI is analyzing your query and searching through meetings..."):
                    try:
                        rag_engine = get_rag_search_engine()
                        rag_results = rag_engine.search(search_query)
                        search_time = time.time() - start_time
                        
                        if rag_results["success"]:
                            st.success(f"Search completed in {search_time:.2f} seconds")
                            
                            # Display the AI answer
                            st.subheader("AI Answer")
                            st.markdown(rag_results["answer"])
                            
                            # Show thought process in an expander
                            with st.expander("View AI reasoning process"):
                                st.markdown(rag_results["thought_process"])
                            # Display snippets in an expander
                            with st.expander("View AI snippets"):
                                st.markdown("\n\n".join(rag_results["snippets"]))
                        else:
                            st.error("AI search failed. Please try another search type or refine your query.")
                    except Exception as e:
                        st.error(f"Error during RAG search: {str(e)}")
            else:
                # Perform keyword or semantic search
                results = search_in_files(
                    search_files, 
                    search_query, 
                    search_in_transcripts=search_in_transcripts,
                    search_in_summaries=search_in_summaries
                )
                search_time = time.time() - start_time
                
                if not results:
                    st.info(f"No results found for '{search_query}'")
                else:
                    st.success(f"Found matches in {len(results)} meeting notes (in {search_time:.2f} seconds)")
                    
                    # Display results
                    for result in results:
                        file_info = result["file_info"]
                        
                        # For semantic search, show the score
                        if st.session_state.search_type == "semantic" and "semantic_score" in result:
                            score_percentage = int(result["semantic_score"] * 100)
                            expander_title = f"{file_info['display_name']} ({result['match_count']} matches, {score_percentage}% relevance)"
                        else:
                            expander_title = f"{file_info['display_name']} ({result['match_count']} matches)"
                        
                        with st.expander(expander_title):
                            # Add tabs for summary and transcript matches
                            if result["summary_matches"] and result["transcript_matches"]:
                                summary_tab, transcript_tab = st.tabs(["Summary Matches", "Transcript Matches"])
                                
                                with summary_tab:
                                    for i, match in enumerate(result["summary_matches"]):
                                        st.markdown(f"**Match {i+1}:**")
                                        
                                        # For semantic search, show the score for each match
                                        if st.session_state.search_type == "semantic" and "semantic_score" in match:
                                            score_percentage = int(match["semantic_score"] * 100)
                                            st.caption(f"Relevance: {score_percentage}%")
                                        
                                        # For keyword search, highlight the search term
                                        if st.session_state.search_type == "keyword":
                                            highlighted_context = re.sub(
                                                f"({search_query})", 
                                                r"**\1**", 
                                                match["context"], 
                                                flags=re.IGNORECASE
                                            )
                                            st.markdown(highlighted_context)
                                        else:
                                            # For semantic search, just show the context
                                            st.markdown(match["context"])
                                        
                                        st.divider()
                                
                                with transcript_tab:
                                    for i, match in enumerate(result["transcript_matches"]):
                                        st.markdown(f"**Match {i+1}:**")
                                        
                                        # For semantic search, show the score for each match
                                        if st.session_state.search_type == "semantic" and "semantic_score" in match:
                                            score_percentage = int(match["semantic_score"] * 100)
                                            st.caption(f"Relevance: {score_percentage}%")
                                        
                                        # For keyword search, highlight the search term
                                        if st.session_state.search_type == "keyword":
                                            highlighted_context = re.sub(
                                                f"({search_query})", 
                                                r"**\1**", 
                                                match["context"], 
                                                flags=re.IGNORECASE
                                            )
                                            st.text(highlighted_context)
                                        else:
                                            # For semantic search, just show the context
                                            st.text(match["context"])
                                        
                                        st.divider()
                            
                            elif result["summary_matches"]:
                                for i, match in enumerate(result["summary_matches"]):
                                    st.markdown(f"**Summary Match {i+1}:**")
                                    
                                    # For semantic search, show the score for each match
                                    if st.session_state.search_type == "semantic" and "semantic_score" in match:
                                        score_percentage = int(match["semantic_score"] * 100)
                                        st.caption(f"Relevance: {score_percentage}%")
                                    
                                    # For keyword search, highlight the search term
                                    if st.session_state.search_type == "keyword":
                                        highlighted_context = re.sub(
                                            f"({search_query})", 
                                            r"**\1**", 
                                            match["context"], 
                                            flags=re.IGNORECASE
                                        )
                                        st.markdown(highlighted_context)
                                    else:
                                        # For semantic search, just show the context
                                        st.markdown(match["context"])
                                    
                                    st.divider()
                            
                            elif result["transcript_matches"]:
                                for i, match in enumerate(result["transcript_matches"]):
                                    st.markdown(f"**Transcript Match {i+1}:**")
                                    
                                    # For semantic search, show the score for each match
                                    if st.session_state.search_type == "semantic" and "semantic_score" in match:
                                        score_percentage = int(match["semantic_score"] * 100)
                                        st.caption(f"Relevance: {score_percentage}%")
                                    
                                    # For keyword search, highlight the search term
                                    if st.session_state.search_type == "keyword":
                                        highlighted_context = re.sub(
                                            f"({search_query})", 
                                            r"**\1**", 
                                            match["context"], 
                                            flags=re.IGNORECASE
                                        )
                                        st.text(highlighted_context)
                                    else:
                                        # For semantic search, just show the context
                                        st.text(match["context"])
                                    st.divider()

def display_analytics():
    """Display analytics about meeting notes."""
    st.header("📊 Meeting Analytics")
    
    # Load meeting files
    meeting_files = load_meeting_files()
    
    if not meeting_files:
        st.warning("No meeting notes found. Please check the meeting notes directory.")
        return
    
    # Calculate statistics
    total_meetings = len(meeting_files)
    
    # Parse all files to get more data
    meeting_data = []
    for file_info in meeting_files:
        content = parse_markdown_file(file_info["path"])
        
        # Calculate word counts
        transcript_word_count = len(content["transcript"].split())
        summary_word_count = len(content["summary"].split())
        
        # Extract action items (lines starting with "- " or "* " after "Action Items" heading)
        action_items = []
        in_action_items = False
        for line in content["summary"].split('\n'):
            if "## Action Items" in line:
                in_action_items = True
                continue
            if in_action_items and line.strip() and (line.strip().startswith('- ') or line.strip().startswith('* ')):
                action_items.append(line.strip())
            elif in_action_items and line.strip() and line.strip().startswith('##'):
                in_action_items = False
        
        meeting_data.append({
            "file_info": file_info,
            "transcript_word_count": transcript_word_count,
            "summary_word_count": summary_word_count,
            "action_items_count": len(action_items),
            "action_items": action_items
        })
    
    # Display overall statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Meetings", total_meetings)
    
    with col2:
        avg_transcript_words = sum(data["transcript_word_count"] for data in meeting_data) / total_meetings if total_meetings > 0 else 0
        st.metric("Avg. Transcript Length", f"{int(avg_transcript_words)} words")
    
    with col3:
        avg_action_items = sum(data["action_items_count"] for data in meeting_data) / total_meetings if total_meetings > 0 else 0
        st.metric("Avg. Action Items", f"{avg_action_items:.1f}")
    
    # Create a timeline of meetings
    st.subheader("Meeting Timeline")
    
    # Prepare data for timeline chart
    timeline_data = pd.DataFrame([
        {
            "date": file_info["date"].strftime("%Y-%m-%d"),
            "meeting": file_info["display_name"],
            "word_count": data["transcript_word_count"],
            "action_items": data["action_items_count"]
        }
        for file_info, data in zip([d["file_info"] for d in meeting_data], meeting_data)
    ])
    
    # Group by date and count meetings
    timeline_grouped = timeline_data.groupby("date").count()[["meeting"]].reset_index()
    timeline_grouped.columns = ["date", "count"]
    
    # Display timeline chart
    st.bar_chart(timeline_grouped.set_index("date")["count"])
    
    # Display meeting length distribution
    st.subheader("Meeting Length Distribution")
    
    # Create bins for word counts
    bins = [0, 1000, 2000, 5000, 10000, float('inf')]
    labels = ['< 1K words', '1K-2K words', '2K-5K words', '5K-10K words', '> 10K words']
    
    # Categorize meetings by length
    transcript_lengths = pd.DataFrame([
        {"word_count": data["transcript_word_count"]}
        for data in meeting_data
    ])
    
    transcript_lengths['length_category'] = pd.cut(
        transcript_lengths['word_count'], 
        bins=bins, 
        labels=labels, 
        right=False
    )
    
    length_distribution = transcript_lengths['length_category'].value_counts().sort_index()
    
    # Display as a bar chart
    st.bar_chart(length_distribution)
    
    # Display action items from recent meetings
    st.subheader("Recent Action Items")
    
    # Sort by date (newest first) and take top 5
    recent_meetings = sorted(meeting_data, key=lambda x: x["file_info"]["date"], reverse=True)[:5]
    
    for meeting in recent_meetings:
        if meeting["action_items"]:
            with st.expander(f"{meeting['file_info']['display_name']} ({len(meeting['action_items'])} action items)"):
                for item in meeting["action_items"]:
                    st.markdown(item)
        
    # Display word cloud of common terms (if wordcloud package is available)
    try:
        from wordcloud import WordCloud
        import matplotlib.pyplot as plt
        
        st.subheader("Common Terms in Meeting Summaries")
        
        # Combine all summaries
        all_summaries = " ".join([parse_markdown_file(file["path"])["summary"] for file in meeting_files])
        
        # Generate word cloud
        wordcloud = WordCloud(width=800, height=400, background_color='white', max_words=100).generate(all_summaries)
        
        # Display the word cloud
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)
    except ImportError:
        st.info("Install the 'wordcloud' package to see a word cloud visualization of common terms.")

def main():
    """Main function to run the Streamlit app."""
    # Add sidebar for navigation
    st.sidebar.title("Meeting Notes Explorer")
    
    # Navigation
    page = st.sidebar.radio("Navigation", ["Meeting Viewer", "Search", "Chat", "Analytics", "Settings"])
    
    if page == "Meeting Viewer":
        display_meeting_viewer()
    elif page == "Search":
        display_search_interface()
    elif page == "Chat":
        display_chat_interface()
    elif page == "Analytics":
        display_analytics()
    else:  # Settings
        display_settings()

def display_settings():
    """Display settings page."""
    st.header("⚙️ Settings")
    
    # Semantic search settings
    st.subheader("Semantic Search Settings")
    
    # Index management
    st.write("Index Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Rebuild Search Index"):
            meeting_files = load_meeting_files()
            with st.spinner("Rebuilding search index..."):
                try:
                    search_engine = get_search_engine()
                    search_engine.reindex_all(meeting_files)
                    st.success("Search index rebuilt successfully!")
                except Exception as e:
                    st.error(f"Failed to rebuild index: {str(e)}")
    
    with col2:
        if st.button("Clear Search Index"):
            with st.spinner("Clearing search index..."):
                try:
                    # Create empty indices
                    search_engine = get_search_engine()
                    search_engine.summary_index = faiss.IndexFlatL2(search_engine.summary_index.d)
                    search_engine.transcript_index = faiss.IndexFlatL2(search_engine.transcript_index.d)
                    search_engine.metadata = {}
                    search_engine.save_indices()
                    st.success("Search index cleared successfully!")
                except Exception as e:
                    st.error(f"Failed to clear index: {str(e)}")
    
    # Display index statistics
    try:
        search_engine = get_search_engine()
        st.subheader("Index Statistics")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Indexed Summaries", search_engine.summary_index.ntotal if search_engine.summary_index else 0)
        with col2:
            st.metric("Indexed Transcript Chunks", search_engine.transcript_index.ntotal if search_engine.transcript_index else 0)
        
        # Show indexed files
        if search_engine.metadata:
            st.subheader("Indexed Files")
            indexed_files = []
            for file_id, meta in search_engine.metadata.items():
                indexed_files.append({
                    "File": meta.get("display_name", os.path.basename(file_id)),
                    "Last Indexed": datetime.fromtimestamp(meta.get("last_modified", 0)).strftime("%Y-%m-%d %H:%M:%S"),
                    "Summary": "✓" if meta.get("summary_idx_active", False) else "✗",
                    "Transcript": "✓" if any(meta.get("transcript_indices_active", [])) else "✗"
                })
            
            # Convert to DataFrame for display
            if indexed_files:
                df = pd.DataFrame(indexed_files)
                st.dataframe(df)
            else:
                st.info("No files have been indexed yet.")
        else:
            st.info("No files have been indexed yet.")
    
    except Exception as e:
        st.error(f"Failed to load index statistics: {str(e)}")


def display_chat_interface():
    """Display the conversational chat interface."""
    st.header("💬 Chat with Meeting Notes")
    
    # Initialize chat history in session state if it doesn't exist
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Load meeting files for filtering
    meeting_files = load_meeting_files()
    
    if not meeting_files:
        st.warning("No meeting notes found. Please check the meeting notes directory.")
        return
    
    # Sidebar for chat settings
    with st.sidebar:
        st.subheader("Chat Settings")
        
        # Date range filter
        st.write("Filter by Date")
        
        # Get min and max dates from files
        dates = [file["date"] for file in meeting_files]
        min_date = min(dates).date() if dates else datetime.now().date()
        max_date = max(dates).date() if dates else datetime.now().date()
        
        start_date = st.date_input("From", min_date, min_value=min_date, max_value=max_date, key="chat_start_date")
        end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date, key="chat_end_date")
        
        # Filter files by date
        filtered_files = [
            file for file in meeting_files 
            if start_date <= file["date"].date() <= end_date
        ]
        
        # Allow selecting specific meetings
        st.write("Select Meetings")
        
        # Create a dataframe for better display
        df = pd.DataFrame([
            {"Meeting": file["display_name"], "Date": file["date"].strftime("%Y-%m-%d %H:%M")}
            for file in filtered_files
        ])
        
        # Display as a table with selection
        selection = st.multiselect(
            "Select specific meetings (leave empty to use all)",
            options=df["Meeting"].tolist(),
            default=[],
            key="chat_meeting_selection"
        )
        
        # Filter files based on selection
        if selection:
            search_files = [file for file in filtered_files if file["display_name"] in selection]
        else:
            search_files = filtered_files
        
        st.caption(f"Using {len(search_files)} meeting notes as context")
        
        # Add button to clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about your meetings...")
    
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Display assistant response with a spinner
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get RAG search engine
                    rag_engine = get_rag_search_engine()
                    
                    # Format chat history for the RAG engine
                    formatted_history = []
                    for msg in st.session_state.chat_history[:-1]:  # Exclude the current message
                        formatted_history.append((msg["role"], msg["content"]))
                    
                    # Perform RAG search with chat history context
                    response = rag_engine.search_with_history(user_input, formatted_history)
                    
                    if response["success"]:
                        answer = response["answer"]
                        st.markdown(answer)
                        
                        # Show thought process and snippets in expanders
                        with st.expander("View reasoning process"):
                            st.markdown(response["thought_process"])
                        
                        with st.expander("View source snippets"):
                            if isinstance(response["snippets"], list):
                                st.markdown("\n\n".join(response["snippets"]))
                            else:
                                st.markdown(response["snippets"])
                    else:
                        error_message = "I'm sorry, I couldn't process your request. Please try again or rephrase your question."
                        st.error(error_message)
                        answer = error_message
                except Exception as e:
                    error_message = f"Error: {str(e)}"
                    st.error(error_message)
                    answer = f"I'm sorry, an error occurred: {str(e)}"
                
                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": answer})

# Run the app
if __name__ == "__main__":
    main()