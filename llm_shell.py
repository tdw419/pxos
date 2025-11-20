"""
Interactive shell for LLM-driven pxOS development
"""

from substrate_enhanced import EnhancedSubstrate

class LLMDevelopmentShell:
    def __init__(self):
        self.substrate = EnhancedSubstrate()
        self.running = True

        # Set up LLM vision regions
        self.substrate.add_llm_vision_region(0, 0, 500, 500)  # Watch main area
        self.substrate.add_llm_vision_region(600, 0, 400, 400)  # Watch code area

    def run_interactive_shell(self):
        """Run interactive LLM development shell"""
        print("🧠 pxOS LLM Development Shell")
        print("Commands:")
        print("  create <description> - LLM creates pixel program")
        print("  analyze - LLM analyzes current VRAM")
        print("  assist - Continuous LLM assistance")
        print("  quit - Exit shell")

        while self.running:
            try:
                command = input("\npxOS-LLM> ").strip()

                if command.startswith("create "):
                    prompt = command[7:]
                    result = self.substrate.llm_think_and_create(prompt)
                    print(f"✅ Created: {result['thinking'][:200]}...")

                elif command == "analyze":
                    analysis = self.substrate.llm.think_about_vram(
                        "Analyze the current computational patterns and suggest optimizations",
                        self.substrate.llm_regions
                    )
                    print(f"🧠 Analysis: {analysis}")

                elif command == "assist":
                    print("🔄 Starting continuous LLM assistance...")
                    # Run in background thread in real implementation
                    self.substrate.continuous_llm_assist()

                elif command == "quit":
                    self.running = False

                else:
                    print("❌ Unknown command")

            except KeyboardInterrupt:
                self.running = False
            except Exception as e:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    shell = LLMDevelopmentShell()
    shell.run_interactive_shell()
