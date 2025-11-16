"""
Storage backends for pixel data and metadata
"""

import hashlib
import os
import sqlite3
import zstandard as zstd
from pathlib import Path
from typing import Optional
import json


class ContentStore:
    """Content-addressable storage for file data (Git-like)"""

    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.objects_dir = self.base_path / "objects"
        self.objects_dir.mkdir(parents=True, exist_ok=True)

    def _hash_content(self, content: bytes) -> str:
        """Calculate SHA-256 hash of content"""
        return hashlib.sha256(content).hexdigest()

    def _object_path(self, hash_str: str) -> Path:
        """Convert hash to filesystem path (Git-style: ab/cdef123...)"""
        return self.objects_dir / hash_str[:2] / hash_str[2:]

    def store(self, content: bytes, compress: bool = True) -> str:
        """
        Store content and return its hash.
        Automatically deduplicates - same content = same hash.
        """
        # Calculate hash
        hash_str = self._hash_content(content)

        # Check if already exists
        obj_path = self._object_path(hash_str)
        if obj_path.exists():
            return hash_str

        # Compress if requested
        if compress:
            compressor = zstd.ZstdCompressor(level=3)
            content = compressor.compress(content)

        # Store content
        obj_path.parent.mkdir(parents=True, exist_ok=True)
        obj_path.write_bytes(content)

        return hash_str

    def retrieve(self, hash_str: str, compressed: bool = True) -> Optional[bytes]:
        """Retrieve content by hash"""
        obj_path = self._object_path(hash_str)

        if not obj_path.exists():
            return None

        content = obj_path.read_bytes()

        # Decompress if needed
        if compressed:
            decompressor = zstd.ZstdDecompressor()
            content = decompressor.decompress(content)

        return content

    def exists(self, hash_str: str) -> bool:
        """Check if hash exists in store"""
        return self._object_path(hash_str).exists()

    def size_on_disk(self, hash_str: str) -> int:
        """Get compressed size on disk"""
        obj_path = self._object_path(hash_str)
        return obj_path.stat().st_size if obj_path.exists() else 0


class MetadataStore:
    """SQLite-based metadata and spatial index"""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self):
        """Initialize database schema"""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS pixels (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                chunk_x INTEGER NOT NULL,
                chunk_y INTEGER NOT NULL,
                visual_r INTEGER,
                visual_g INTEGER,
                visual_b INTEGER,
                visual_a INTEGER,
                name TEXT NOT NULL,
                path TEXT NOT NULL UNIQUE,
                size INTEGER,
                permissions TEXT,
                owner INTEGER,
                grp INTEGER,
                created TEXT,
                modified TEXT,
                accessed TEXT,
                mime_type TEXT,
                data_hash TEXT NOT NULL,
                backend TEXT,
                compressed INTEGER,
                compression TEXT,
                parent_id TEXT,
                FOREIGN KEY (parent_id) REFERENCES pixels(id)
            );

            CREATE INDEX IF NOT EXISTS idx_position ON pixels(x, y);
            CREATE INDEX IF NOT EXISTS idx_chunk ON pixels(chunk_x, chunk_y);
            CREATE INDEX IF NOT EXISTS idx_path ON pixels(path);
            CREATE INDEX IF NOT EXISTS idx_hash ON pixels(data_hash);
            CREATE INDEX IF NOT EXISTS idx_parent ON pixels(parent_id);

            CREATE TABLE IF NOT EXISTS directory_children (
                parent_id TEXT NOT NULL,
                child_id TEXT NOT NULL,
                PRIMARY KEY (parent_id, child_id),
                FOREIGN KEY (parent_id) REFERENCES pixels(id),
                FOREIGN KEY (child_id) REFERENCES pixels(id)
            );

            CREATE INDEX IF NOT EXISTS idx_dir_parent ON directory_children(parent_id);
            CREATE INDEX IF NOT EXISTS idx_dir_child ON directory_children(child_id);
        """)
        self.conn.commit()

    def store_pixel(self, pixel: "Pixel"):
        """Store or update pixel metadata"""
        from .pixel import Pixel

        self.conn.execute("""
            INSERT OR REPLACE INTO pixels (
                id, type, x, y, chunk_x, chunk_y,
                visual_r, visual_g, visual_b, visual_a,
                name, path, size, permissions, owner, grp,
                created, modified, accessed, mime_type,
                data_hash, backend, compressed, compression, parent_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pixel.id, pixel.type.value,
            pixel.position.x, pixel.position.y,
            pixel.position.chunk_id[0], pixel.position.chunk_id[1],
            pixel.visual.r, pixel.visual.g, pixel.visual.b, pixel.visual.a,
            pixel.metadata.name, pixel.metadata.path, pixel.metadata.size,
            pixel.metadata.permissions, pixel.metadata.owner, pixel.metadata.group,
            pixel.metadata.created.isoformat(),
            pixel.metadata.modified.isoformat(),
            pixel.metadata.accessed.isoformat(),
            pixel.metadata.mime_type,
            pixel.data.hash, pixel.data.backend,
            1 if pixel.data.compressed else 0, pixel.data.compression,
            pixel.parent_id
        ))

        # Update directory children relationships
        if pixel.children:
            for child_id in pixel.children:
                self.conn.execute("""
                    INSERT OR IGNORE INTO directory_children (parent_id, child_id)
                    VALUES (?, ?)
                """, (pixel.id, child_id))

        self.conn.commit()

    def get_pixel_by_path(self, path: str) -> Optional[dict]:
        """Retrieve pixel metadata by file path"""
        cursor = self.conn.execute("""
            SELECT * FROM pixels WHERE path = ?
        """, (path,))

        row = cursor.fetchone()
        if not row:
            return None

        return self._row_to_dict(row)

    def get_pixel_by_id(self, pixel_id: str) -> Optional[dict]:
        """Retrieve pixel metadata by ID"""
        cursor = self.conn.execute("""
            SELECT * FROM pixels WHERE id = ?
        """, (pixel_id,))

        row = cursor.fetchone()
        if not row:
            return None

        return self._row_to_dict(row)

    def get_pixels_in_chunk(self, chunk_x: int, chunk_y: int) -> list[dict]:
        """Get all pixels in a 256x256 chunk"""
        cursor = self.conn.execute("""
            SELECT * FROM pixels WHERE chunk_x = ? AND chunk_y = ?
        """, (chunk_x, chunk_y))

        return [self._row_to_dict(row) for row in cursor.fetchall()]

    def get_children(self, parent_id: str) -> list[str]:
        """Get child pixel IDs for a directory"""
        cursor = self.conn.execute("""
            SELECT child_id FROM directory_children WHERE parent_id = ?
        """, (parent_id,))

        return [row[0] for row in cursor.fetchall()]

    def _row_to_dict(self, row) -> dict:
        """Convert database row to dictionary"""
        return {
            "id": row[0],
            "type": row[1],
            "position": {"x": row[2], "y": row[3]},
            "chunk_id": (row[4], row[5]),
            "visual": {"r": row[6], "g": row[7], "b": row[8], "a": row[9]},
            "metadata": {
                "name": row[10],
                "path": row[11],
                "size": row[12],
                "permissions": row[13],
                "owner": row[14],
                "group": row[15],
                "created": row[16],
                "modified": row[17],
                "accessed": row[18],
                "mime_type": row[19]
            },
            "data": {
                "hash": row[20],
                "backend": row[21],
                "compressed": bool(row[22]),
                "compression": row[23]
            },
            "parent_id": row[24]
        }

    def get_stats(self) -> dict:
        """Get storage statistics"""
        cursor = self.conn.execute("""
            SELECT
                COUNT(*) as total_pixels,
                COUNT(DISTINCT data_hash) as unique_files,
                SUM(size) as total_size,
                COUNT(CASE WHEN type = 'directory' THEN 1 END) as directories,
                COUNT(CASE WHEN type = 'file' THEN 1 END) as files
            FROM pixels
        """)

        row = cursor.fetchone()
        return {
            "total_pixels": row[0],
            "unique_files": row[1],
            "total_size": row[2] or 0,
            "directories": row[3] or 0,
            "files": row[4] or 0
        }

    def close(self):
        """Close database connection"""
        self.conn.close()
