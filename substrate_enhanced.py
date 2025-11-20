"""
Substrate with built-in LLM vision capabilities
"""
from substrate import Substrate
from llm_in_vram import VRAMEmbeddedLLM

class EnhancedSubstrate(Substrate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.llm = VRAMEmbeddedLLM(self)
        self.llm_regions = []  # Regions LLM is "watching"

    def add_llm_vision_region(self, x: int, y: int, width: int, height: int):
        """Add a VRAM region for LLM to monitor"""
        self.llm_regions.append((x, y, width, height))

    def llm_think_and_create(self, prompt: str):
        """Let LLM analyze VRAM and create new patterns"""
        return self.llm.execute_llm_directive(prompt)

    def continuous_llm_assist(self):
        """Run LLM in continuous assistance mode"""
        print("🧠 LLM Assistant Mode Active")
        print("   LLM will watch VRAM and suggest improvements")

        while True:
            try:
                # Let LLM analyze current state
                analysis = self.llm.think_about_vram(
                    "Analyze the current VRAM state and suggest one improvement",
                    self.llm_regions
                )

                print(f"🧠 LLM Analysis: {analysis[:100]}...")

                # Implement suggestion if it makes sense
                if "wire" in analysis.lower() or "clock" in analysis.lower():
                    self.llm.llm_to_vram_pattern(analysis, 200, 200)

            except Exception as e:
                print(f"❌ LLM assist error: {e}")

            # Don't spam - think every 10 seconds
            import time
            time.sleep(10)
