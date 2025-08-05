"""
Simple caching system for shared prompts to organize them into Discord threads
"""

import sqlite3
import hashlib
from typing import Optional, Tuple
from datetime import datetime
import json

class ShareCache:
    def __init__(self, db_path: str = "share_cache.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Initialize the SQLite database for tracking shared prompts"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS shared_prompts (
                prompt_hash TEXT PRIMARY KEY,
                prompt_text TEXT NOT NULL,
                channel_id TEXT NOT NULL,
                thread_id TEXT,
                share_count INTEGER DEFAULT 1,
                first_shared TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_shared TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_channel_prompt 
            ON shared_prompts(channel_id, prompt_hash)
        ''')
        
        conn.commit()
        conn.close()
    
    def get_prompt_hash(self, prompt: str) -> str:
        """Generate a simple hash for the prompt (normalized)"""
        # Normalize: lowercase, strip whitespace, remove extra spaces
        normalized = ' '.join(prompt.lower().strip().split())
        # Use first 12 chars of hash for readability
        return hashlib.sha256(normalized.encode()).hexdigest()[:12]
    
    def check_similar_prompt(self, prompt: str, channel_id: str, window_hours: int = 24) -> Tuple[bool, Optional[str]]:
        """
        Check if a similar prompt was recently shared in this channel
        Returns: (is_similar, thread_id)
        """
        prompt_hash = self.get_prompt_hash(prompt)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Auto-cleanup old entries on each check (lightweight operation)
        cursor.execute('''
            DELETE FROM shared_prompts 
            WHERE datetime(last_shared) < datetime('now', '-' || ? || ' hours')
        ''', (window_hours,))
        
        # Look for exact or very similar prompt in same channel within window
        cursor.execute('''
            SELECT thread_id, share_count, prompt_text
            FROM shared_prompts 
            WHERE channel_id = ? AND prompt_hash = ?
            AND datetime(last_shared) > datetime('now', '-' || ? || ' hours')
        ''', (channel_id, prompt_hash, window_hours))
        
        result = cursor.fetchone()
        conn.commit()
        conn.close()
        
        if result:
            return (True, result[0])  # Similar prompt found, return thread_id if exists
        
        return (False, None)
    
    def record_share(self, prompt: str, channel_id: str, thread_id: Optional[str] = None) -> str:
        """Record that a prompt was shared"""
        prompt_hash = self.get_prompt_hash(prompt)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if already exists
        cursor.execute('''
            SELECT share_count FROM shared_prompts 
            WHERE prompt_hash = ? AND channel_id = ?
        ''', (prompt_hash, channel_id))
        
        existing = cursor.fetchone()
        
        if existing:
            # Update existing record
            cursor.execute('''
                UPDATE shared_prompts 
                SET share_count = share_count + 1,
                    last_shared = CURRENT_TIMESTAMP,
                    thread_id = COALESCE(?, thread_id)
                WHERE prompt_hash = ? AND channel_id = ?
            ''', (thread_id, prompt_hash, channel_id))
        else:
            # Insert new record
            cursor.execute('''
                INSERT INTO shared_prompts 
                (prompt_hash, prompt_text, channel_id, thread_id)
                VALUES (?, ?, ?, ?)
            ''', (prompt_hash, prompt[:500], channel_id, thread_id))
        
        conn.commit()
        conn.close()
        
        return prompt_hash
    
    def update_thread_id(self, prompt_hash: str, channel_id: str, thread_id: str):
        """Update the thread ID for a prompt hash"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE shared_prompts 
            SET thread_id = ?
            WHERE prompt_hash = ? AND channel_id = ?
        ''', (thread_id, prompt_hash, channel_id))
        
        conn.commit()
        conn.close()
    
    def cleanup_old_entries(self, days: int = 30):
        """Remove entries older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            DELETE FROM shared_prompts 
            WHERE last_shared < datetime('now', '-' || ? || ' days')
        ''', (days,))
        
        deleted = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted
    
    def get_channel_stats(self, channel_id: str, window_hours: int = 24) -> dict:
        """Get sharing statistics for a channel within time window"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Clean up old entries first
        cursor.execute('''
            DELETE FROM shared_prompts 
            WHERE datetime(last_shared) < datetime('now', '-' || ? || ' hours')
        ''', (window_hours,))
        
        cursor.execute('''
            SELECT 
                COUNT(DISTINCT prompt_hash) as unique_prompts,
                SUM(share_count) as total_shares,
                COUNT(DISTINCT thread_id) as thread_count
            FROM shared_prompts
            WHERE channel_id = ?
            AND datetime(last_shared) > datetime('now', '-' || ? || ' hours')
        ''', (channel_id, window_hours))
        
        result = cursor.fetchone()
        conn.commit()
        conn.close()
        
        return {
            'unique_prompts': result[0] or 0,
            'total_shares': result[1] or 0,
            'threads_created': result[2] or 0,
            'window_hours': window_hours
        }

# Global instance
share_cache = ShareCache()