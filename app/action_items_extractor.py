import os
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("action-items-extractor")

class ActionItemsExtractor:
    """
    A tool to extract action items and next steps from meeting notes.
    """
    
    def __init__(self):
        """Initialize the action items extractor."""
        pass
    
    def extract_from_file(self, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract action items from a single meeting file.
        
        Args:
            file_info: Dictionary containing file information including path and content
            
        Returns:
            List of dictionaries containing action items with metadata
        """
        action_items = []
        
        try:
            file_path = file_info.get("path", None)
            if not file_path:
                logger.error("No file path provided")
                return []
            
            with open(file_path, 'r') as f:
                content = f.read()

            # Split content into summary and transcript
            summary_match = re.search(r'# Meeting Summary(.*?)(?=# Transcript|$)', content, re.DOTALL)
            transcript_match = re.search(r'# Transcript(.*?)$', content, re.DOTALL)
            
            summary_text = summary_match.group(1).strip() if summary_match else ""
            transcript_text = transcript_match.group(1).strip() if transcript_match else ""

            # Extract from summary (more structured)
            if summary_text:
                summary_items = self._extract_from_summary(summary_text, file_info)
                action_items.extend(summary_items)
            
            # Extract from transcript (less structured, but might contain additional items)
            if transcript_text:
                transcript_items = self._extract_from_transcript(transcript_text, file_info)
                action_items.extend(transcript_items)
            
            return action_items
            
        except Exception as e:
            logger.error(f"Error extracting action items from {file_info.get('display_name', 'unknown file')}: {str(e)}")
            return []
    
    def _extract_from_summary(self, summary: str, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract action items from the meeting summary.
        
        Args:
            summary: The meeting summary text
            file_info: Dictionary containing file metadata
            
        Returns:
            List of action items found in the summary
        """
        action_items = []
        
        # Look for action items section in the summary
        action_section_patterns = [
            r"(?:^|\n)#+\s*Action\s*Items?.*?(?=\n#+|$)",
            r"(?:^|\n)#+\s*Next\s*Steps?.*?(?=\n#+|$)",
            r"(?:^|\n)#+\s*To-?Do.*?(?=\n#+|$)",
            r"(?:^|\n)#+\s*Follow-?ups?.*?(?=\n#+|$)"
        ]
        
        for pattern in action_section_patterns:
            matches = re.findall(pattern, summary, re.IGNORECASE | re.DOTALL)
            for match in matches:
                # Extract bullet points or numbered items
                items = re.findall(r"(?:^|\n)(?:[-*•]|\d+\.)\s*(.*?)(?=\n[-*•]|\n\d+\.|\n#+|$)", match, re.DOTALL)
                
                for item in items:
                    item_text = item.strip()
                    if item_text:
                        # Try to extract assignee if present (common formats like "John: do something" or "John to do something")
                        assignee_match = re.match(r"([^:]+):\s*(.*)", item_text) or re.match(r"([^:]+)\s+to\s+(.*)", item_text)
                        
                        assignee = None
                        if assignee_match:
                            assignee = assignee_match.group(1).strip()
                            item_text = assignee_match.group(2).strip()
                        
                        action_items.append({
                            "text": item_text,
                            "assignee": assignee,
                            "source": "summary",
                            "meeting_name": file_info.get("display_name", ""),
                            "meeting_date": file_info.get("date", datetime.now()),
                            "file_path": file_info.get("path", "")
                        })
        
        # If no structured action items found, look for bullet points or numbered lists in the entire summary
        if not action_items:
            items = re.findall(r"(?:^|\n)(?:[-*•]|\d+\.)\s*(.*?)(?=\n[-*•]|\n\d+\.|\n#+|$)", summary, re.DOTALL)
            
            for item in items:
                item_text = item.strip()
                # Only include if it looks like an action item (contains action verbs)
                if self._is_likely_action_item(item_text):
                    # Try to extract assignee if present
                    assignee_match = re.match(r"([^:]+):\s*(.*)", item_text) or re.match(r"([^:]+)\s+to\s+(.*)", item_text)
                    
                    assignee = None
                    if assignee_match:
                        assignee = assignee_match.group(1).strip()
                        item_text = assignee_match.group(2).strip()
                    
                    action_items.append({
                        "text": item_text,
                        "assignee": assignee,
                        "source": "summary",
                        "meeting_name": file_info.get("display_name", ""),
                        "meeting_date": file_info.get("date", datetime.now()),
                        "file_path": file_info.get("path", "")
                    })
        
        return action_items
    
    def _extract_from_transcript(self, transcript: str, file_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract action items from the meeting transcript.
        
        Args:
            transcript: The meeting transcript text
            file_info: Dictionary containing file metadata
            
        Returns:
            List of action items found in the transcript
        """
        action_items = []
        
        # Look for phrases that typically indicate action items in transcripts
        action_phrases = [
            r"(?:need|needs|needed) to (\w+)",
            r"(?:will|should|must|going to) (\w+)",
            r"(?:action item|next step|follow-?up|to-?do)(?:s)?(?: is| are)? (?:to )?(\w+)",
            r"(?:let's|let us) (\w+)",
            r"(?:I'll|I will|we'll|we will) (\w+)",
            r"(?:assigned to|responsible for) ([^\.]+)"
        ]
        
        for pattern in action_phrases:
            # Find sentences containing action phrases
            sentences = re.split(r'(?<=[.!?])\s+', transcript)
            for sentence in sentences:
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Try to extract assignee
                    assignee = None
                    assignee_patterns = [
                        r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?) (?:will|should|is going to)",
                        r"([A-Z][a-z]+(?:\s[A-Z][a-z]+)?) (?:is|are) responsible for",
                        r"assigned to ([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)"
                    ]
                    
                    for assignee_pattern in assignee_patterns:
                        assignee_match = re.search(assignee_pattern, sentence)
                        if assignee_match:
                            assignee = assignee_match.group(1)
                            break
                    
                    # Only add if it's not already in the list (avoid duplicates)
                    item_text = sentence.strip()
                    if not any(item["text"] == item_text for item in action_items):
                        action_items.append({
                            "text": item_text,
                            "assignee": assignee,
                            "source": "transcript",
                            "meeting_name": file_info.get("display_name", ""),
                            "meeting_date": file_info.get("date", datetime.now()),
                            "file_path": file_info.get("path", "")
                        })
        
        return action_items
    
    def _is_likely_action_item(self, text: str) -> bool:
        """
        Check if the given text is likely to be an action item.
        
        Args:
            text: The text to check
            
        Returns:
            True if the text is likely an action item, False otherwise
        """
        # Common action verbs
        action_verbs = [
            "create", "update", "review", "complete", "finish", "implement", "develop",
            "prepare", "schedule", "organize", "contact", "follow up", "investigate",
            "research", "send", "share", "write", "document", "present", "discuss"
        ]
        
        # Check if the text contains any action verbs
        for verb in action_verbs:
            if re.search(r'\b' + verb + r'\b', text, re.IGNORECASE):
                return True
        
        # Check for common action item patterns
        action_patterns = [
            r"\bwill\b", r"\bshould\b", r"\bmust\b", r"\bneed(s|ed)?\b",
            r"\bto-?do\b", r"\btask\b", r"\baction\b", r"\bnext step\b"
        ]
        
        for pattern in action_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        
        return False
    
    def extract_from_files(self, files: List[Dict[str, Any]], start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> str:
        """
        Extract action items from multiple meeting files within a date range.
        
        Args:
            files: List of file information dictionaries
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            
        Returns:
            List of action items from all files within the date range
        """
        all_action_items = []
        
        # Filter files by date if specified
        filtered_files = files
        if start_date or end_date:
            filtered_files = []
            for file in files:
                file_date = file.get("date")
                if file_date:
                    if start_date and file_date.date() < start_date.date():
                        continue
                    if end_date and file_date.date() > end_date.date():
                        continue
                filtered_files.append(file)
        
        # Extract action items from each file
        for file in filtered_files:
            items = self.extract_from_file(file)
            all_action_items.extend(items)
        
        return self.format_action_items(all_action_items)
    
    def format_action_items(self, action_items: List[Dict[str, Any]]) -> str:
        """
        Format action items into a readable string.
        
        Args:
            action_items: List of action item dictionaries
            
        Returns:
            Formatted string of action items
        """
        if not action_items:
            return "No action items found."
        
        # Group action items by meeting
        meeting_items = {}
        for item in action_items:
            meeting_name = item["meeting_name"]
            meeting_date = item["meeting_date"]
            
            key = f"{meeting_name} ({meeting_date.strftime('%Y-%m-%d')})"
            if key not in meeting_items:
                meeting_items[key] = []
            
            meeting_items[key].append(item)
        
        # Format the output
        output = []
        for meeting, items in meeting_items.items():
            output.append(f"## Action Items from {meeting}")
            
            for i, item in enumerate(items, 1):
                item_text = f"{i}. {item['text']}"
                if item["assignee"]:
                    item_text += f" (Assigned to: {item['assignee']})"
                output.append(item_text)
            
            output.append("")  # Add a blank line between meetings
        
        return "\n".join(output)

# Singleton instance
_extractor_instance = None

def get_action_items_extractor() -> ActionItemsExtractor:
    """Get or create the action items extractor instance."""
    global _extractor_instance
    if _extractor_instance is None:
        _extractor_instance = ActionItemsExtractor()
    return _extractor_instance