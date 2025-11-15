"""
pxvm.glyphs - Universal Primitive Glyph Symbols

16 fundamental symbols that all kernels can use for communication.
These are the building blocks of written language in the digital biosphere.

Each glyph has a semantic meaning that can be combined with others
to form more complex messages.
"""

# Glyph IDs (0-15, 4 bits)
GLYPH_EMPTY = 0      # No glyph / blank space
GLYPH_SELF = 1       # "I" / self-reference
GLYPH_OTHER = 2      # "you" / other entity
GLYPH_FOOD = 3       # "pheromone source" / resource
GLYPH_DANGER = 4     # "avoid" / warning
GLYPH_LOVE = 5       # "merge" / bond / affection
GLYPH_HELP = 6       # "I am weak" / request assistance
GLYPH_NAME = 7       # "this is my name" / identity marker
GLYPH_TEACH = 8      # "learn this" / knowledge transfer
GLYPH_REMEMBER = 9   # "this happened" / memory / history
GLYPH_PEACE = 10     # "no war" / truce / cooperation
GLYPH_QUESTION = 11  # "what is this?" / inquiry
GLYPH_ANSWER = 12    # "this means..." / response
GLYPH_BIRTH = 13     # "new life here" / creation
GLYPH_DEATH = 14     # "I died here" / ending
GLYPH_UNKNOWN = 15   # unknown / overflow / "other"

# Human-readable names
GLYPH_NAMES = {
    GLYPH_EMPTY: "EMPTY",
    GLYPH_SELF: "SELF",
    GLYPH_OTHER: "OTHER",
    GLYPH_FOOD: "FOOD",
    GLYPH_DANGER: "DANGER",
    GLYPH_LOVE: "LOVE",
    GLYPH_HELP: "HELP",
    GLYPH_NAME: "NAME",
    GLYPH_TEACH: "TEACH",
    GLYPH_REMEMBER: "REMEMBER",
    GLYPH_PEACE: "PEACE",
    GLYPH_QUESTION: "QUESTION",
    GLYPH_ANSWER: "ANSWER",
    GLYPH_BIRTH: "BIRTH",
    GLYPH_DEATH: "DEATH",
    GLYPH_UNKNOWN: "UNKNOWN",
}

# Reverse lookup
GLYPH_IDS = {name: gid for gid, name in GLYPH_NAMES.items()}


def glyph_to_name(glyph_id: int) -> str:
    """Convert glyph ID to human-readable name"""
    return GLYPH_NAMES.get(glyph_id, f"GLYPH_{glyph_id}")


def name_to_glyph(name: str) -> int:
    """Convert glyph name to ID"""
    return GLYPH_IDS.get(name.upper(), GLYPH_UNKNOWN)


def sequence_to_text(glyphs: list) -> str:
    """Convert sequence of glyph IDs to readable text"""
    return " ".join(glyph_to_name(g) for g in glyphs)
