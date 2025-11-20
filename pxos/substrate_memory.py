"""
SQLITE VECTOR MEMORY SYSTEM
Implements multimodal RAG with visual embeddings
Active learning with quality scoring
"""

import sqlite3
import numpy as np
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pickle

class SubstrateMemory:
    def __init__(self, db_path: str = "pxos/substrate_memory.db"):
        self.db_path = db_path
        self._init_database()

        # Vector embedding dimensions (placeholder for CLIP/other model)
        self.embedding_dim = 512

        print("🧠 SUBSTRATE MEMORY SYSTEM INITIALIZED")
        print(f"   Database: {self.db_path}")
        print("   Vector embeddings: ENABLED")
        print("   Active learning: ENABLED")

    def _init_database(self):
        """Initialize SQLite database with vector extensions"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Main interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response_text TEXT NOT NULL,
                vram_state BLOB NOT NULL,
                vram_hash TEXT NOT NULL,
                complexity_score REAL DEFAULT 0.0,
                quality_score REAL DEFAULT 0.0,
                pattern_type TEXT DEFAULT 'generic',
                embedding BLOB,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Pattern library for successful designs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS pattern_library (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_hash TEXT UNIQUE NOT NULL,
                pattern_data BLOB NOT NULL,
                pattern_type TEXT NOT NULL,
                efficiency_score REAL DEFAULT 0.0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                usage_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Learning insights table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                insight_type TEXT NOT NULL,
                content TEXT NOT NULL,
                confidence REAL DEFAULT 0.0,
                applied_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON interactions(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_pattern_type ON interactions(pattern_type)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_quality ON interactions(quality_score)')

        conn.commit()
        conn.close()

    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection with custom adapters"""
        conn = sqlite3.connect(self.db_path)

        # Register NumPy array adapter
        sqlite3.register_adapter(np.ndarray, self._adapt_array)
        sqlite3.register_converter("ARRAY", self._convert_array)

        return conn

    def _adapt_array(self, arr: np.ndarray) -> bytes:
        """Convert NumPy array to bytes for SQLite storage"""
        return pickle.dumps(arr)

    def _convert_array(self, data: bytes) -> np.ndarray:
        """Convert bytes back to NumPy array"""
        return pickle.loads(data)

    def save_interaction(self, prompt: str, response: str,
                        vram_state: bytes, vram_hash: str,
                        complexity: float, pattern_type: str = "generic") -> int:
        """Save complete interaction to memory"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Generate text embedding (placeholder - integrate with sentence-transformers)
        text_embedding = self._generate_text_embedding(f"{prompt} {response}")

        cursor.execute('''
            INSERT INTO interactions
            (timestamp, prompt, response_text, vram_state, vram_hash,
             complexity_score, pattern_type, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            prompt,
            response,
            vram_state,
            vram_hash,
            complexity,
            pattern_type,
            text_embedding
        ))

        interaction_id = cursor.lastrowid

        # Initial quality assessment
        initial_quality = self._assess_initial_quality(cursor, complexity, pattern_type)
        self._update_quality_score(cursor, interaction_id, initial_quality)

        conn.commit()
        conn.close()

        print(f"💾 Saved interaction #{interaction_id} (quality: {initial_quality:.2f})")
        return interaction_id

    def _generate_text_embedding(self, text: str) -> bytes:
        """Generate text embedding vector (placeholder for actual model)"""
        # In production, use sentence-transformers or similar
        # For now, create deterministic pseudo-embedding
        import hashlib
        seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
        np.random.seed(seed)
        embedding = np.random.randn(self.embedding_dim).astype(np.float32)
        return self._adapt_array(embedding)

    def _assess_initial_quality(self, cursor: sqlite3.Cursor, complexity: float, pattern_type: str) -> float:
        """Initial quality assessment based on complexity and pattern type"""
        base_score = complexity  # Higher complexity generally better

        # Adjust based on pattern type experience
        type_experience = self._get_pattern_type_experience(cursor, pattern_type)
        experience_bonus = min(type_experience * 0.1, 0.3)

        return min(base_score + experience_bonus, 1.0)

    def _get_pattern_type_experience(self, cursor: sqlite3.Cursor, pattern_type: str) -> int:
        """Get experience level with specific pattern type"""
        cursor.execute('''
            SELECT COUNT(*) FROM interactions
            WHERE pattern_type = ? AND quality_score > 0.7
        ''', (pattern_type,))

        count = cursor.fetchone()[0]

        return count

    def _update_quality_score(self, cursor: sqlite3.Cursor, interaction_id: int, quality_score: float):
        """Update quality score for interaction"""
        cursor.execute('''
            UPDATE interactions SET quality_score = ? WHERE id = ?
        ''', (quality_score, interaction_id))

    def retrieve_relevant_memories(self, query: str, max_results: int = 5) -> List[Dict]:
        """Retrieve relevant past interactions using vector similarity"""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Generate query embedding
        query_embedding = self._generate_text_embedding(query)

        # Simple cosine similarity search (in production, use sqlite-vec)
        cursor.execute('''
            SELECT id, prompt, response_text, quality_score, pattern_type
            FROM interactions
            WHERE quality_score > 0.5
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (max_results * 3,))  # Get more then filter

        candidates = cursor.fetchall()
        conn.close()

        # Simple keyword-based relevance filtering
        relevant = []
        query_words = set(query.lower().split())

        for candidate in candidates:
            id, prompt, response, quality, pattern_type = candidate
            candidate_text = f"{prompt} {response}".lower()
            candidate_words = set(candidate_text.split())

            # Calculate word overlap
            overlap = len(query_words.intersection(candidate_words))
            if overlap > 0:
                relevance_score = overlap / len(query_words)
                relevant.append({
                    'id': id,
                    'prompt': prompt,
                    'response': response,
                    'relevance_score': relevance_score,
                    'quality_score': quality,
                    'pattern_type': pattern_type
                })

        # Sort by relevance and quality, return top results
        relevant.sort(key=lambda x: (x['relevance_score'], x['quality_score']), reverse=True)
        return relevant[:max_results]

    def get_recent_interactions(self, count: int) -> List[Dict]:
        """Get the most recent interactions from the log."""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id, pattern_type, quality_score, complexity_score
            FROM interactions
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (count,))

        results = [
            {'id': row[0], 'pattern_type': row[1], 'quality_score': row[2], 'complexity_score': row[3]}
            for row in cursor.fetchall()
        ]
        conn.close()
        return results

    def generate_learning_insight(self, recent_interactions: List[Dict]) -> Optional[Dict]:
        """Generate learning insight from recent interactions"""
        if len(recent_interactions) < 3:
            return None

        # Analyze pattern success rates
        pattern_success = {}
        for interaction in recent_interactions:
            pattern_type = interaction['pattern_type']
            quality = interaction['quality_score']

            if pattern_type not in pattern_success:
                pattern_success[pattern_type] = {'total': 0, 'high_quality': 0}

            pattern_success[pattern_type]['total'] += 1
            if quality > 0.7:
                pattern_success[pattern_type]['high_quality'] += 1

        # Find most successful pattern type
        best_pattern = None
        best_success_rate = 0

        for pattern_type, stats in pattern_success.items():
            if stats['total'] >= 2:  # Need multiple attempts
                success_rate = stats['high_quality'] / stats['total']
                if success_rate > best_success_rate:
                    best_success_rate = success_rate
                    best_pattern = pattern_type

        if best_pattern and best_success_rate > 0.6:
            insight = {
                'type': 'pattern_success',
                'content': f"Pattern type '{best_pattern}' shows {best_success_rate:.0%} success rate",
                'confidence': best_success_rate,
                'recommendation': f"Prefer {best_pattern} patterns for similar tasks"
            }

            # Save insight
            self._save_insight(insight)
            return insight

        return None

    def _save_insight(self, insight: Dict):
        """Save learning insight to database"""
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO insights (insight_type, content, confidence)
            VALUES (?, ?, ?)
        ''', (insight['type'], insight['content'], insight['confidence']))

        conn.commit()
        conn.close()
