import torch
from typing import List, Dict, Any, Optional
import logging
from sentence_transformers import CrossEncoder
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meeting-reranker")

class MeetingNotesReRanker:
    """Specialized re-ranker for meeting notes content."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"):
        # switch to L-6 variant or ms-marco-MiniLM-L-12-v2 for higher quality but slower inference
        """Initialize the meeting notes re-ranker."""
        try:
            self.cross_encoder = CrossEncoder(model_name)
            logger.info(f"Initialized meeting notes re-ranker with model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize cross-encoder: {str(e)}")
            # Fall back to keyword-based ranking if model fails
            self.cross_encoder = None
    
    def rerank(self, query: str, results: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Re-rank meeting notes search results with specialized scoring."""
        if not results:
            return []
        
        # Extract all contexts for scoring
        contexts = []
        result_map = []
        
        for result_idx, result in enumerate(results):
            # Process summary matches
            for match_idx, match in enumerate(result.get("summary_matches", [])):
                contexts.append((query, match["context"]))
                result_map.append((result_idx, "summary", match_idx))
            
            # Process transcript matches
            for match_idx, match in enumerate(result.get("transcript_matches", [])):
                contexts.append((query, match["context"]))
                result_map.append((result_idx, "transcript", match_idx))
        
        if not contexts:
            return results[:top_k]
        
        try:
            # Apply cross-encoder scoring if available
            if self.cross_encoder:
                # Get cross-encoder scores
                cross_scores = self.cross_encoder.predict(contexts)
                
                # Apply scores to results
                for i, score in enumerate(cross_scores):
                    result_idx, match_type, match_idx = result_map[i]
                    
                    # Apply additional meeting-specific scoring factors
                    final_score = self._apply_meeting_specific_factors(
                        base_score=float(score),
                        query=query,
                        context=contexts[i][1],
                        match_type=match_type
                    )
                    
                    # Update the score in the results
                    if match_type == "summary":
                        results[result_idx]["summary_matches"][match_idx]["rerank_score"] = final_score
                    else:
                        results[result_idx]["transcript_matches"][match_idx]["rerank_score"] = final_score
            else:
                # Fall back to keyword-based scoring
                for i, (query, context) in enumerate(contexts):
                    result_idx, match_type, match_idx = result_map[i]
                    score = self._keyword_score(query, context)
                    
                    if match_type == "summary":
                        results[result_idx]["summary_matches"][match_idx]["rerank_score"] = score
                    else:
                        results[result_idx]["transcript_matches"][match_idx]["rerank_score"] = score
            
            # Calculate overall result scores
            for result in results:
                all_matches = result.get("summary_matches", []) + result.get("transcript_matches", [])
                result["rerank_score"] = max([match.get("rerank_score", 0) for match in all_matches]) if all_matches else 0
            
            # Sort by the new score
            reranked_results = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
            
            return reranked_results[:top_k]
            
        except Exception as e:
            logger.error(f"Error in meeting notes re-ranking: {str(e)}")
            # Fall back to original ranking
            return results[:top_k]
    
    def _apply_meeting_specific_factors(self, base_score: float, query: str, context: str, match_type: str) -> float:
        """Apply meeting-specific scoring factors."""
        score = base_score
        
        # 1. Boost summaries slightly as they're often more relevant for quick answers
        if match_type == "summary":
            score *= 1.1
        
        # 2. Boost contexts containing speaker identification patterns
        if re.search(r'(?:^|\n)([A-Z][a-z]+(?: [A-Z][a-z]+)*):|\b(?:said|mentioned|noted|explained)\b', context):
            score *= 1.05
        
        # 3. Boost contexts containing action items or decisions
        action_keywords = ['action', 'task', 'todo', 'to-do', 'assigned', 'deadline', 'due', 
                          'decision', 'agreed', 'resolved', 'conclusion']
        if any(keyword in query.lower() for keyword in action_keywords):
            if any(keyword in context.lower() for keyword in action_keywords):
                score *= 1.15
        
        # 4. Boost contexts containing dates or times when query asks about timing
        time_keywords = ['when', 'date', 'time', 'schedule', 'deadline']
        if any(keyword in query.lower() for keyword in time_keywords):
            if re.search(r'\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|january|february|march|april|june|july|august|september|october|november|december)\b|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b|\b\d{1,2}:\d{2}\b', context.lower()):
                score *= 1.1
        
        # 5. Penalize very short contexts slightly
        if len(context.split()) < 10:
            score *= 0.9
        
        return score
    
    def _keyword_score(self, query: str, context: str) -> float:
        """Calculate keyword-based score as fallback."""
        query_terms = set(query.lower().split())
        context_terms = set(context.lower().split())
        
        # Calculate overlap
        overlap = len(query_terms.intersection(context_terms))
        coverage = overlap / max(1, len(query_terms))
        
        # Calculate exact phrase matches
        exact_matches = 0
        query_bigrams = self._get_ngrams(query.lower(), 2)
        context_bigrams = self._get_ngrams(context.lower(), 2)
        bigram_overlap = len(query_bigrams.intersection(context_bigrams))
        
        # Combine scores
        score = (coverage * 0.7) + (bigram_overlap / max(1, len(query_bigrams)) * 0.3)
        return score
    
    def _get_ngrams(self, text: str, n: int) -> set:
        """Generate n-grams from text."""
        words = text.split()
        return set(' '.join(words[i:i+n]) for i in range(len(words)-n+1))