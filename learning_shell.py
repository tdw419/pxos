"""
Learning-Aware VRAM Shell
Tracks and improves LLM performance over time
"""

from learning_llm import LearningEnhancedLLM
from substrate_direct_llm import DirectLLMSubstrate
import numpy as np

class LearningShell:
    def __init__(self):
        self.substrate = DirectLLMSubstrate()
        self.learning_llm = LearningEnhancedLLM(self.substrate)

        print("🎓 Learning-Aware Shell Active")
        print("   LLM improves with each interaction")

        # Show initial learning status
        self._show_learning_status()

    def run_learning_shell(self):
        """Main learning shell loop"""
        while True:
            try:
                user_input = input("\nlearn> ").strip()

                if user_input.lower() in ['exit', 'quit']:
                    break
                elif user_input.lower() == 'status':
                    self._show_learning_status()
                elif user_input.lower() == 'report':
                    self._generate_learning_report()
                elif user_input.lower() == 'patterns':
                    self._show_pattern_library()
                elif user_input.lower().startswith('similar '):
                    self._find_similar_patterns(user_input[8:])
                elif user_input:
                    # Use learning-enhanced LLM
                    result = self.learning_llm.learn_and_respond(user_input)

                    print(f"✅ Learning Response (used {result['learning_used']} previous patterns)")
                    print(f"   Pattern Hash: {result.get('pattern_hash', 'None')}")

                    # Show learning impact
                    if result['learning_used'] > 0:
                        print("   🎓 Applied previous learning")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")

    def _show_learning_status(self):
        """Show current learning status"""
        report = self.learning_llm.generate_learning_report()

        print(f"\n📊 LEARNING STATUS:")
        print(f"   Patterns: {report['total_patterns']}")
        print(f"   Interactions: {report['total_interactions']}")
        print(f"   Learning Progress: {report['learning_progress']:.1%}")

        if report['top_efficient_patterns']:
            print(f"   Top Pattern: {report['top_efficient_patterns'][0]['type']} "
                  f"(eff: {report['top_efficient_patterns'][0]['efficiency']:.2f})")

    def _generate_learning_report(self):
        """Generate detailed learning report"""
        report = self.learning_llm.generate_learning_report()

        print(f"\n📈 DETAILED LEARNING REPORT")
        print(f"Total Patterns: {report['total_patterns']}")
        print(f"Total Interactions: {report['total_interactions']}")
        print(f"Overall Progress: {report['learning_progress']:.1%}")

        print(f"\n🏆 TOP EFFICIENT PATTERNS:")
        for i, pattern in enumerate(report['top_efficient_patterns'][:3], 1):
            print(f"  {i}. {pattern['type']} - {pattern['purpose']}")
            print(f"     Efficiency: {pattern['efficiency']:.2f}, "
                  f"Successes: {pattern['success_count']}")

        if report['recent_insights']:
            print(f"\n💡 RECENT INSIGHTS:")
            for insight in report['recent_insights'][:3]:
                print(f"  - {insight['content'][:100]}...")

    def _show_pattern_library(self):
        """Show pattern library summary"""
        library = self.learning_llm.learning_system.pattern_library

        print(f"\n📚 PATTERN LIBRARY ({len(library)} patterns)")

        # Group by type
        type_counts = {}
        for pattern_data in library.values():
            pattern_type = pattern_data['type']
            type_counts[pattern_type] = type_counts.get(pattern_type, 0) + 1

        for pattern_type, count in type_counts.items():
            avg_efficiency = np.mean([
                p['efficiency_score']
                for p in library.values()
                if p['type'] == pattern_type
            ])
            print(f"  {pattern_type}: {count} patterns (avg eff: {avg_efficiency:.2f})")

    def _find_similar_patterns(self, prompt: str):
        """Find patterns similar to the current request"""
        # Create a dummy pattern for similarity search
        dummy_pattern = np.zeros((32, 32, 4), dtype=np.float32)
        similar = self.learning_llm.learning_system.find_similar_patterns(dummy_pattern)

        print(f"\n🔍 SIMILAR PATTERNS TO: '{prompt}'")
        for i, pattern in enumerate(similar[:3], 1):
            print(f"  {i}. {pattern['type']} - {pattern['purpose']}")
            print(f"     Similarity: {pattern['similarity']:.2f}, "
                  f"Efficiency: {pattern['efficiency_score']:.2f}")

if __name__ == "__main__":
    shell = LearningShell()
    shell.run_learning_shell()
