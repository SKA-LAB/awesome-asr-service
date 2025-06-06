import os
import logging
from notion_client import Client
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("notion-integration")

class NotionIntegration:
    def __init__(self, token: str = None, database_id: str = None):
        """
        Initialize the Notion integration.
        
        Args:
            token: Notion API integration token
            database_id: ID of the Notion database where meeting notes will be stored
        """
        self.token = token or os.environ.get("NOTION_TOKEN")
        self.database_id = database_id or os.environ.get("NOTION_DATABASE_ID")
        
        if not self.token:
            logger.error("Notion token not provided")
            raise ValueError("Notion token is required. Set NOTION_TOKEN environment variable or pass token parameter.")
        
        if not self.database_id:
            logger.error("Notion database ID not provided")
            raise ValueError("Notion database ID is required. Set NOTION_DATABASE_ID environment variable or pass database_id parameter.")
        
        self.client = Client(auth=self.token)
        logger.info("Notion client initialized")
    
    def create_meeting_page(self, 
                           title: str, 
                           summary: str, 
                           transcript: str, 
                           meeting_date: Optional[datetime] = None,
                           participants: Optional[list] = None) -> Dict[str, Any]:
        """
        Create a new page in Notion with meeting details.
        
        Args:
            title: Title of the meeting
            summary: Markdown formatted summary of the meeting
            transcript: Full transcript of the meeting
            meeting_date: Date of the meeting (defaults to current date)
            participants: List of meeting participants (optional)
            
        Returns:
            Dict containing the response from Notion API
        """
        if meeting_date is None:
            meeting_date = datetime.now()
        
        # Format date for Notion
        formatted_date = meeting_date.strftime("%Y-%m-%d")
        
        # Prepare the page properties
        properties = {
            "Name": {
                "title": [
                    {
                        "text": {
                            "content": title
                        }
                    }
                ]
            },
            "Date": {
                "date": {
                    "start": formatted_date
                }
            }
        }
        
        # Add participants if provided
        if participants:
            properties["Participants"] = {
                "rich_text": [
                    {
                        "text": {
                            "content": ", ".join(participants)
                        }
                    }
                ]
            }
        
        # Notion has a limit of 2000 characters per rich_text block
        MAX_CHUNK_SIZE = 1900  # Slightly less than the limit to be safe
        
        # Prepare the page content
        children = [
            # Meeting Summary Section
            {
                "object": "block",
                "type": "heading_1",
                "heading_1": {
                    "rich_text": [{"type": "text", "text": {"content": "Meeting Summary"}}]
                }
            },
            {
                "object": "block",
                "type": "paragraph",
                "paragraph": {
                    "rich_text": [{"type": "text", "text": {"content": ""}}]
                }
            }
        ]
        
        # Handle the summary - split into chunks if needed
        if len(summary) <= MAX_CHUNK_SIZE:
            # If summary is short enough, use a single callout block
            children.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{"type": "text", "text": {"content": summary}}],
                    "icon": {"emoji": "📝"}
                }
            })
        else:
            # For longer summaries, split into multiple blocks
            # First add an intro callout
            children.append({
                "object": "block",
                "type": "callout",
                "callout": {
                    "rich_text": [{"type": "text", "text": {"content": "Meeting Summary (continued below)"}}],
                    "icon": {"emoji": "📝"}
                }
            })
            
            # Split summary into chunks
            summary_chunks = []
            for i in range(0, len(summary), MAX_CHUNK_SIZE):
                summary_chunks.append(summary[i:i + MAX_CHUNK_SIZE])
            
            # Add each chunk as a paragraph block
            for chunk in summary_chunks:
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": chunk}}]
                    }
                })
        
        # Add separator
        children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": ""}}]
            }
        })
        
        # Add transcript heading
        children.append({
            "object": "block",
            "type": "heading_1",
            "heading_1": {
                "rich_text": [{"type": "text", "text": {"content": "Full Transcript"}}]
            }
        })
        
        children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": ""}}]
            }
        })
        
        # Handle the transcript - split into chunks if needed
        # If transcript is short enough, use toggle block
        if len(transcript) <= MAX_CHUNK_SIZE:
            children.append({
                "object": "block",
                "type": "toggle",
                "toggle": {
                    "rich_text": [{"type": "text", "text": {"content": "Expand to view full transcript"}}],
                    "children": [
                        {
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {
                                "rich_text": [{"type": "text", "text": {"content": transcript}}]
                            }
                        }
                    ]
                }
            })
        else:
            # For longer transcripts, split into multiple blocks under a toggle
            chunks = []
            for i in range(0, len(transcript), MAX_CHUNK_SIZE):
                chunks.append(transcript[i:i + MAX_CHUNK_SIZE])
            
            # Create paragraph blocks for each chunk
            chunk_blocks = []
            for chunk in chunks:
                chunk_blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": chunk}}]
                    }
                })
            
            # Add a toggle containing all the chunks
            children.append({
                "object": "block",
                "type": "toggle",
                "toggle": {
                    "rich_text": [{"type": "text", "text": {"content": "Expand to view full transcript"}}],
                    "children": chunk_blocks
                }
            })
        
        try:
            # Create the page in Notion
            logger.info(f"Creating Notion page for meeting: {title}")
            response = self.client.pages.create(
                parent={"database_id": self.database_id},
                properties=properties,
                children=children
            )
            
            logger.info(f"Successfully created Notion page with ID: {response['id']}")
            return response
        
        except Exception as e:
            logger.error(f"Failed to create Notion page: {str(e)}")
            raise
    
    def test_connection(self) -> bool:
        """Test the connection to Notion API"""
        try:
            # Try to retrieve the database to verify credentials
            self.client.databases.retrieve(self.database_id)
            logger.info("Successfully connected to Notion API")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Notion API: {str(e)}")
            return False


# Example usage
if __name__ == "__main__":
    # This will run only when the script is executed directly
    try:
        # Get token and database ID from environment variables
        token = os.environ.get("NOTION_TOKEN")
        database_id = os.environ.get("NOTION_DATABASE_ID")
        
        if not token or not database_id:
            print("Please set NOTION_TOKEN and NOTION_DATABASE_ID environment variables")
            exit(1)
        
        # Initialize the Notion integration
        notion = NotionIntegration(token, database_id)
        
        # Test the connection
        if notion.test_connection():
            print("Connection to Notion successful!")
            
            # Create a test page
            response = notion.create_meeting_page(
                title="Test Meeting",
                summary="## Meeting Summary\n\nThis is a test summary.\n\n## Action Items\n\n- Task 1\n- Task 2",
                transcript="This is a test transcript of the meeting.",
                participants=["John Doe", "Jane Smith"]
            )
            
            print(f"Test page created with ID: {response['id']}")
            print(f"URL: {response['url']}")
        else:
            print("Failed to connect to Notion")
    
    except Exception as e:
        print(f"Error: {str(e)}")