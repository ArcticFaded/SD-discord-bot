"""
Prompt caching and similarity system using sentence embeddings and Discord threads
"""

import sqlite3
import hashlib
import json
import numpy as np
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import asyncio
from dataclasses import dataclass, asdict
from sentence_transformers import SentenceTransformer
import pickle

@dataclass
class CachedPrompt:
    prompt_hash: str
    prompt_text: str
    embedding: bytes  # Pickled numpy array
    thread_id: Optional[str]
    channel_id: str
    message_id: Optional[str]
    image_url: Optional[str]
    parameters: str  # JSON string
    created_at: str
    hit_count: int
    last_accessed: str

class PromptCacheManager:
    def __init__(self, db_path: str = "prompt_cache.db", similarity_threshold: float = 0.85):
        self.db_path = db_path
        self.similarity_threshold = similarity_threshold
        self.model = None  # Lazy load the model
        self._init_db()
    
    def _get_model(self):
        """Lazy load the sentence transformer model"""
        if self.model is None:
            # Using a lightweight model for fast similarity checks
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
        return self.model
    
    def _init_db(self):
        """Initialize the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Main cache table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS prompt_cache (
                prompt_hash TEXT PRIMARY KEY,
                prompt_text TEXT NOT NULL,
                embedding BLOB NOT NULL,
                thread_id TEXT,
                channel_id TEXT NOT NULL,
                message_id TEXT,
                image_url TEXT,
                parameters TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                hit_count INTEGER DEFAULT 0,
                last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Thread management table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS thread_groups (
                thread_id TEXT PRIMARY KEY,
                channel_id TEXT NOT NULL,
                base_prompt TEXT NOT NULL,
                prompt_count INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Similarity index for faster lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_channel_thread 
            ON prompt_cache(channel_id, thread_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_prompt(self, prompt: str, parameters: Optional[Dict] = None) -> str:
        """Generate a hash for prompt + key parameters"""
        # Include important parameters in hash (size, sampler, etc)
        hash_input = prompt.lower().strip()
        
        if parameters:
            # Only include parameters that affect the image
            key_params = {
                'width': parameters.get('width'),
                'height': parameters.get('height'),
                'sampler_index': parameters.get('sampler_index'),
                'cfg_scale': parameters.get('cfg_scale')
            }
            hash_input += json.dumps(key_params, sort_keys=True)
        
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get sentence embedding for similarity comparison"""
        model = self._get_model()
        return model.encode(text, convert_to_numpy=True)
    
    def cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings"""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return dot_product / (norm1 * norm2)
    
    def find_similar_prompts(
        self, 
        prompt: str, 
        channel_id: str, 
        limit: int = 5
    ) -> List[Tuple[CachedPrompt, float]]:
        """Find similar prompts in the cache"""
        embedding = self.get_embedding(prompt)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all prompts from this channel
        cursor.execute('''
            SELECT * FROM prompt_cache 
            WHERE channel_id = ?
            ORDER BY last_accessed DESC
            LIMIT 100
        ''', (channel_id,))
        
        results = []
        for row in cursor.fetchall():
            cached_embedding = pickle.loads(row[2])
            similarity = self.cosine_similarity(embedding, cached_embedding)
            
            if similarity >= self.similarity_threshold:
                cached_prompt = CachedPrompt(
                    prompt_hash=row[0],
                    prompt_text=row[1],
                    embedding=row[2],
                    thread_id=row[3],
                    channel_id=row[4],
                    message_id=row[5],
                    image_url=row[6],
                    parameters=row[7],
                    created_at=row[8],
                    hit_count=row[9],
                    last_accessed=row[10]
                )
                results.append((cached_prompt, similarity))
        
        conn.close()
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def add_to_cache(
        self,
        prompt: str,
        channel_id: str,
        parameters: Dict,
        thread_id: Optional[str] = None,
        message_id: Optional[str] = None,
        image_url: Optional[str] = None
    ) -> str:
        """Add a prompt to the cache"""
        prompt_hash = self.hash_prompt(prompt, parameters)
        embedding = self.get_embedding(prompt)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if already exists
        cursor.execute('SELECT hit_count FROM prompt_cache WHERE prompt_hash = ?', (prompt_hash,))
        existing = cursor.fetchone()
        
        if existing:
            # Update hit count and last accessed
            cursor.execute('''
                UPDATE prompt_cache 
                SET hit_count = hit_count + 1,
                    last_accessed = CURRENT_TIMESTAMP,
                    message_id = COALESCE(?, message_id),
                    image_url = COALESCE(?, image_url)
                WHERE prompt_hash = ?
            ''', (message_id, image_url, prompt_hash))
        else:
            # Insert new entry
            cursor.execute('''
                INSERT INTO prompt_cache 
                (prompt_hash, prompt_text, embedding, thread_id, channel_id, 
                 message_id, image_url, parameters, hit_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            ''', (
                prompt_hash,
                prompt,
                pickle.dumps(embedding),
                thread_id,
                channel_id,
                message_id,
                image_url,
                json.dumps(parameters)
            ))
        
        # Update thread group if provided
        if thread_id:
            cursor.execute('''
                INSERT OR REPLACE INTO thread_groups 
                (thread_id, channel_id, base_prompt, prompt_count, last_updated)
                VALUES (
                    ?,
                    ?,
                    ?,
                    COALESCE((SELECT prompt_count + 1 FROM thread_groups WHERE thread_id = ?), 1),
                    CURRENT_TIMESTAMP
                )
            ''', (thread_id, channel_id, prompt[:100], thread_id))
        
        conn.commit()
        conn.close()
        
        return prompt_hash
    
    def get_cached_result(self, prompt_hash: str) -> Optional[CachedPrompt]:
        """Get a cached result by hash"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM prompt_cache WHERE prompt_hash = ?', (prompt_hash,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return CachedPrompt(
                prompt_hash=row[0],
                prompt_text=row[1],
                embedding=row[2],
                thread_id=row[3],
                channel_id=row[4],
                message_id=row[5],
                image_url=row[6],
                parameters=row[7],
                created_at=row[8],
                hit_count=row[9],
                last_accessed=row[10]
            )
        return None
    
    def should_create_thread(self, similar_prompts: List[Tuple[CachedPrompt, float]]) -> bool:
        """Determine if we should create a new thread for this group"""
        if not similar_prompts:
            return False
        
        # Check if we have enough similar prompts without a thread
        prompts_without_thread = [p for p, _ in similar_prompts if p.thread_id is None]
        
        # Create thread if we have 3+ similar prompts or 2+ without a thread
        return len(similar_prompts) >= 3 or len(prompts_without_thread) >= 2
    
    def get_thread_for_prompt(self, prompt: str, channel_id: str) -> Optional[str]:
        """Get the best thread for a prompt based on similarity"""
        similar = self.find_similar_prompts(prompt, channel_id, limit=1)
        
        if similar and similar[0][1] >= 0.9:  # Very high similarity
            return similar[0][0].thread_id
        
        return None
    
    def update_thread_association(self, prompt_hashes: List[str], thread_id: str):
        """Update multiple prompts to be associated with a thread"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.executemany(
            'UPDATE prompt_cache SET thread_id = ? WHERE prompt_hash = ?',
            [(thread_id, h) for h in prompt_hashes]
        )
        
        conn.commit()
        conn.close()
    
    def get_cache_stats(self, channel_id: Optional[str] = None) -> Dict:
        """Get cache statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if channel_id:
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_prompts,
                    SUM(hit_count) as total_hits,
                    COUNT(DISTINCT thread_id) as thread_count
                FROM prompt_cache
                WHERE channel_id = ?
            ''', (channel_id,))
        else:
            cursor.execute('''
                SELECT 
                    COUNT(*) as total_prompts,
                    SUM(hit_count) as total_hits,
                    COUNT(DISTINCT thread_id) as thread_count
                FROM prompt_cache
            ''')
        
        result = cursor.fetchone()
        conn.close()
        
        return {
            'total_prompts': result[0] or 0,
            'total_hits': result[1] or 0,
            'thread_count': result[2] or 0,
            'cache_efficiency': (result[1] or 0) / max(1, result[0])
        }
    
    def cleanup_old_entries(self, days: int = 30):
        """Remove entries older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM prompt_cache 
            WHERE last_accessed < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted

# Global cache manager instance
cache_manager = PromptCacheManager()