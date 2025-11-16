"""
Core pixel data structures following pxOS specification v0.1
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List
import json
import uuid
from datetime import datetime


class PixelType(Enum):
    FILE = "file"
    DIRECTORY = "directory"
    EXECUTABLE = "executable"


@dataclass
class Position:
    """Absolute position on infinite 2D map"""
    x: int
    y: int

    @property
    def chunk_id(self) -> tuple[int, int]:
        """Calculate which 256x256 chunk this position belongs to"""
        return (self.x // 256, self.y // 256)

    def to_dict(self) -> dict:
        return {
            "x": self.x,
            "y": self.y,
            "chunk_id": f"{self.chunk_id[0]},{self.chunk_id[1]}"
        }


@dataclass
class PixelVisual:
    """Visual representation of pixel"""
    r: int  # 0-255
    g: int  # 0-255
    b: int  # 0-255
    a: int = 255  # Alpha channel
    size: int = 1  # Pixel radius/dimensions

    def to_dict(self) -> dict:
        return {
            "r": self.r,
            "g": self.g,
            "b": self.b,
            "a": self.a,
            "size": self.size
        }

    @staticmethod
    def from_file_properties(mime_type: str, size: int, access_time: float) -> "PixelVisual":
        """Generate visual encoding based on file properties"""
        # R channel: File type
        r = PixelVisual._encode_file_type(mime_type)

        # G channel: Size class (logarithmic)
        g = PixelVisual._encode_size(size)

        # B channel: Access recency
        b = PixelVisual._encode_access_time(access_time)

        return PixelVisual(r=r, g=g, b=b)

    @staticmethod
    def _encode_file_type(mime_type: str) -> int:
        """Map MIME type to R channel value"""
        if not mime_type:
            return 0

        mime = mime_type.lower()
        if "executable" in mime or "x-executable" in mime:
            return 255
        elif mime.startswith("image/") or mime.startswith("video/") or mime.startswith("audio/"):
            return 200
        elif mime.startswith("text/"):
            return 150
        elif "zip" in mime or "tar" in mime or "compress" in mime:
            return 100
        elif "config" in mime or mime == "application/json" or mime == "application/xml":
            return 50
        else:
            return 0

    @staticmethod
    def _encode_size(size: int) -> int:
        """Map file size to G channel (logarithmic scale)"""
        if size == 0:
            return 0
        elif size > 1_000_000_000:  # > 1GB
            return 255
        elif size > 100_000_000:  # 100MB - 1GB
            return 200
        elif size > 10_000_000:  # 10MB - 100MB
            return 150
        elif size > 1_000_000:  # 1MB - 10MB
            return 100
        else:  # < 1MB
            return 50

    @staticmethod
    def _encode_access_time(access_time: float) -> int:
        """Map access recency to B channel"""
        now = datetime.now().timestamp()
        age = now - access_time

        if age < 86400:  # < 24 hours
            return 255
        elif age < 604800:  # < 1 week
            return 128
        else:
            return 0


@dataclass
class Metadata:
    """File/directory metadata"""
    name: str
    path: str
    size: int
    permissions: str  # Octal string like "0644"
    owner: int
    group: int
    created: datetime
    modified: datetime
    accessed: datetime
    mime_type: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": self.path,
            "size": self.size,
            "permissions": self.permissions,
            "owner": self.owner,
            "group": self.group,
            "created": self.created.isoformat(),
            "modified": self.modified.isoformat(),
            "accessed": self.accessed.isoformat(),
            "mime_type": self.mime_type
        }


@dataclass
class DataReference:
    """Reference to actual file data in content store"""
    hash: str  # SHA-256 hash
    backend: str  # "local", "lancedb", etc.
    compressed: bool = True
    compression: str = "zstd"

    def to_dict(self) -> dict:
        return {
            "hash": self.hash,
            "backend": self.backend,
            "compressed": self.compressed,
            "compression": self.compression
        }


@dataclass
class Pixel:
    """Main pixel structure - represents one file or directory"""
    type: PixelType
    position: Position
    visual: PixelVisual
    metadata: Metadata
    data: DataReference
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    children: List[str] = field(default_factory=list)  # Child pixel IDs
    parent_id: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict per spec"""
        return {
            "$schema": "pxos-pixel-v0.1",
            "type": self.type.value,
            "id": self.id,
            "position": self.position.to_dict(),
            "visual": self.visual.to_dict(),
            "metadata": self.metadata.to_dict(),
            "data": self.data.to_dict(),
            "children": self.children,
            "parent_id": self.parent_id
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string"""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "Pixel":
        """Deserialize from dict"""
        return cls(
            type=PixelType(data["type"]),
            id=data["id"],
            position=Position(x=data["position"]["x"], y=data["position"]["y"]),
            visual=PixelVisual(**{k: v for k, v in data["visual"].items() if k != "size"}),
            metadata=Metadata(
                name=data["metadata"]["name"],
                path=data["metadata"]["path"],
                size=data["metadata"]["size"],
                permissions=data["metadata"]["permissions"],
                owner=data["metadata"]["owner"],
                group=data["metadata"]["group"],
                created=datetime.fromisoformat(data["metadata"]["created"]),
                modified=datetime.fromisoformat(data["metadata"]["modified"]),
                accessed=datetime.fromisoformat(data["metadata"]["accessed"]),
                mime_type=data["metadata"].get("mime_type")
            ),
            data=DataReference(**data["data"]),
            children=data.get("children", []),
            parent_id=data.get("parent_id")
        )
