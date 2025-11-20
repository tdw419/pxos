"""
VRAM Learning System
Saves all LLM interactions and enables continuous improvement
"""

import numpy as np
import json
import pickle
import sqlite3
from datetime import datetime
from typing import Dict, List, Any, Tuple
import hashlib

class VRAMLearningSystem:
    def __init__(self, substrate, db_path="llm_memory.db"):
        self.substrate = substrate
        self.db_path = db_path

        # Learning memory regions in VRAM
        self.learning_memory_region = (2000, 0, 1024, 1024)
        self.pattern_library_region = (2000, 1024, 1024, 1024)
        self.feedback_region = (3000, 2048, 512, 512)

        # Initialize learning database
        self._init_learning_db()

        # Load existing knowledge
        self.pattern_library = self._load_pattern_library()
        self.interaction_history = self._load_recent_interactions()

        print("🧠 VRAM Learning System Initialized")
        print(f"   Pattern Library: {len(self.pattern_library)} patterns")
        print(f"   Interaction History: {len(self.interaction_history)} entries")

    def _init_learning_db(self):
        """Initialize SQLite database for learning storage"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                vram_region TEXT NOT NULL,
                pattern_hash TEXT NOT NULL,
                success_score REAL DEFAULT 0.0,
                learning_notes TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        # Patterns table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_hash TEXT UNIQUE NOT NULL,
                pattern_data BLOB NOT NULL,
                pattern_type TEXT NOT NULL,
                coordinates TEXT NOT NULL,
                purpose TEXT NOT NULL,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                efficiency_score REAL DEFAULT 0.0,
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

        conn.commit()
        conn.close()

    def save_interaction(self, prompt: str, response: str,
                        vram_region: Tuple[int, int, int, int],
                        pattern: np.ndarray) -> str:
        """Save an interaction to learning database"""
        # Generate pattern hash
        pattern_hash = self._hash_pattern(pattern)

        # Store in database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO interactions
            (timestamp, prompt, response, vram_region, pattern_hash)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            prompt,
            response,
            json.dumps(vram_region),
            pattern_hash
        ))

        # Also store pattern if new
        if pattern_hash not in self.pattern_library:
            self._save_pattern(conn, cursor, pattern_hash, pattern, vram_region, prompt)

        interaction_id = cursor.lastrowid
        conn.commit()
        conn.close()

        # Update in-memory cache
        self.interaction_history.append({
            'id': interaction_id,
            'prompt': prompt,
            'response': response,
            'pattern_hash': pattern_hash,
            'timestamp': datetime.now()
        })

        # Update VRAM learning display
        self._update_learning_display()

        return pattern_hash

    def _save_pattern(self, conn: sqlite3.Connection, cursor: sqlite3.Cursor,
                     pattern_hash: str, pattern: np.ndarray,
                     coordinates: Tuple, purpose: str):
        """Save a pattern to the pattern library"""
        # Serialize pattern data
        pattern_data = pickle.dumps(pattern)

        cursor.execute('''
            INSERT OR IGNORE INTO patterns
            (pattern_hash, pattern_data, pattern_type, coordinates, purpose)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            pattern_hash,
            pattern_data,
            self._classify_pattern(pattern),
            json.dumps(coordinates),
            purpose
        ))

        # Update pattern library cache
        self.pattern_library[pattern_hash] = {
            'pattern': pattern,
            'type': self._classify_pattern(pattern),
            'purpose': purpose,
            'success_count': 0,
            'efficiency_score': 0.0
        }

    def record_success(self, pattern_hash: str, efficiency_metrics: Dict[str, float]):
        """Record successful pattern execution"""
        if pattern_hash in self.pattern_library:
            self.pattern_library[pattern_hash]['success_count'] += 1

            # Calculate efficiency score
            efficiency_score = self._calculate_efficiency_score(efficiency_metrics)
            self.pattern_library[pattern_hash]['efficiency_score'] = efficiency_score

            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE patterns
                SET success_count = success_count + 1,
                    efficiency_score = ?
                WHERE pattern_hash = ?
            ''', (efficiency_score, pattern_hash))

            conn.commit()
            conn.close()

    def record_failure(self, pattern_hash: str, error_info: str):
        """Record pattern failure"""
        if pattern_hash in self.pattern_library:
            # Update database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                UPDATE patterns
                SET failure_count = failure_count + 1
                WHERE pattern_hash = ?
            ''', (pattern_hash,))

            # Save failure insight
            cursor.execute('''
                INSERT INTO insights (insight_type, content, confidence)
                VALUES (?, ?, ?)
            ''', (
                'pattern_failure',
                f"Pattern {pattern_hash} failed: {error_info}",
                0.8
            ))

            conn.commit()
            conn.close()

    def _hash_pattern(self, pattern: np.ndarray) -> str:
        """Generate unique hash for a pixel pattern"""
        return hashlib.sha256(pattern.tobytes()).hexdigest()[:16]

    def _classify_pattern(self, pattern: np.ndarray) -> str:
        """Classify pattern type for learning"""
        # Analyze pattern characteristics
        wire_ratio = np.sum(pattern[:, :, 0] == 3.0) / pattern.size
        head_ratio = np.sum(pattern[:, :, 0] == 1.0) / pattern.size
        structure_score = self._calculate_structure_score(pattern)

        if wire_ratio > 0.6:
            return "dense_circuit"
        elif head_ratio > 0.1:
            return "active_computation"
        elif structure_score > 0.7:
            return "structured_logic"
        else:
            return "emergent_pattern"

    def _calculate_structure_score(self, pattern: np.ndarray) -> float:
        """Calculate how structured/organized a pattern is"""
        # Simple structure detection
        from scipy import ndimage

        binary_pattern = pattern[:, :, 0] > 0.1
        if np.sum(binary_pattern) == 0:
            return 0.0

        # Calculate connectivity and regularity
        labeled, num_features = ndimage.label(binary_pattern)
        feature_sizes = ndimage.sum(binary_pattern, labeled, range(1, num_features + 1))

        if len(feature_sizes) == 0:
            return 0.0

        size_std = np.std(feature_sizes)
        max_size = np.max(feature_sizes)

        # More uniform feature sizes = more structured
        structure_score = 1.0 - (size_std / max_size if max_size > 0 else 0)
        return float(structure_score)

    def _calculate_efficiency_score(self, metrics: Dict[str, float]) -> float:
        """Calculate overall efficiency score from metrics"""
        weights = {
            'execution_speed': 0.3,
            'memory_usage': 0.2,
            'pattern_density': 0.2,
            'computational_power': 0.3
        }

        score = 0.0
        for metric, weight in weights.items():
            if metric in metrics:
                score += metrics[metric] * weight

        return min(score, 1.0)

    def _load_pattern_library(self) -> Dict[str, Any]:
        """Load pattern library from database"""
        pattern_library = {}

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT pattern_hash, pattern_data, pattern_type, purpose,
                       success_count, efficiency_score
                FROM patterns
            ''')

            for row in cursor.fetchall():
                pattern_hash, pattern_data, pattern_type, purpose, success_count, efficiency_score = row
                pattern = pickle.loads(pattern_data)

                pattern_library[pattern_hash] = {
                    'pattern': pattern,
                    'type': pattern_type,
                    'purpose': purpose,
                    'success_count': success_count,
                    'efficiency_score': efficiency_score
                }

            conn.close()
        except:
            pass  # First run, database might not exist yet

        return pattern_library

    def _load_recent_interactions(self, limit: int = 100) -> List[Dict]:
        """Load recent interactions from database"""
        interactions = []

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                SELECT id, timestamp, prompt, response, pattern_hash
                FROM interactions
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))

            for row in cursor.fetchall():
                id, timestamp, prompt, response, pattern_hash = row
                interactions.append({
                    'id': id,
                    'timestamp': datetime.fromisoformat(timestamp),
                    'prompt': prompt,
                    'response': response,
                    'pattern_hash': pattern_hash
                })

            conn.close()
        except:
            pass

        return interactions

    def get_learning_insights(self) -> List[Dict]:
        """Get recent learning insights"""
        insights = []

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT insight_type, content, confidence, applied_count
            FROM insights
            ORDER BY created_at DESC
            LIMIT 10
        ''')

        for row in cursor.fetchall():
            insight_type, content, confidence, applied_count = row
            insights.append({
                'type': insight_type,
                'content': content,
                'confidence': confidence,
                'applied_count': applied_count
            })

        conn.close()
        return insights

    def find_similar_patterns(self, target_pattern: np.ndarray,
                             max_results: int = 5) -> List[Dict]:
        """Find similar patterns in the library"""
        target_hash = self._hash_pattern(target_pattern)
        similarities = []

        for pattern_hash, pattern_data in self.pattern_library.items():
            if pattern_hash == target_hash:
                continue  # Skip exact matches

            similarity = self._pattern_similarity(target_pattern, pattern_data['pattern'])
            if similarity > 0.3:  # Only reasonably similar patterns
                similarities.append({
                    'pattern_hash': pattern_hash,
                    'similarity': similarity,
                    'type': pattern_data['type'],
                    'purpose': pattern_data['purpose'],
                    'success_count': pattern_data['success_count'],
                    'efficiency_score': pattern_data['efficiency_score']
                })

        # Sort by similarity and return top results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:max_results]

    def _pattern_similarity(self, pattern1: np.ndarray, pattern2: np.ndarray) -> float:
        """Calculate similarity between two patterns"""
        # Ensure same dimensions
        min_shape = (min(pattern1.shape[0], pattern2.shape[0]),
                    min(pattern1.shape[1], pattern2.shape[1]))

        p1_slice = pattern1[:min_shape[0], :min_shape[1], 0]  # Use only state channel
        p2_slice = pattern2[:min_shape[0], :min_shape[1], 0]

        # Simple similarity based on state matching
        matches = np.sum(p1_slice == p2_slice)
        total_pixels = p1_slice.size

        return matches / total_pixels if total_pixels > 0 else 0.0

    def _update_learning_display(self):
        """Update VRAM learning status display"""
        from vram_text_display import VRAMTextDisplay
        text_display = VRAMTextDisplay(self.substrate)

        stats_text = f"""
LEARNING STATS:
Patterns: {len(self.pattern_library)}
Interactions: {len(self.interaction_history)}
Successful: {sum(1 for p in self.pattern_library.values() if p['success_count'] > 0)}
Avg Efficiency: {np.mean([p['efficiency_score'] for p in self.pattern_library.values()]):.2f}
        """.strip()

        text_display.render_text_to_vram(
            stats_text,
            self.learning_memory_region[0] + 8,
            self.learning_memory_region[1] + 8,
            color=(0.0, 1.0, 0.0, 1.0),  # Green text
            background=(0.0, 0.0, 0.0, 1.0)
        )
