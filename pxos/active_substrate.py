"""
ACTIVE LEARNING SUBSTRATE LLM
Implements the complete cognitive loop with RAG and self-improvement
"""

from .substrate_display import SubstrateDisplay
from .substrate_memory import SubstrateMemory
import requests
import json
from typing import Dict, Any, List
import requests

class ActiveSubstrateLLM:
    def __init__(self, lmstudio_host: str = "http://localhost:1234"):
        self.display = SubstrateDisplay()
        self.memory = SubstrateMemory()
        self.lmstudio_host = lmstudio_host

        # Learning state
        self.interaction_count = 0
        self.learning_progress = 0.0
        self.recent_insights = []

        print("🎯 ACTIVE SUBSTRATE LLM INITIALIZED")
        print("   VRAM-native display: READY")
        print("   Vector memory: READY")
        print("   Active learning: ENABLED")
        print("   Cognitive loop: CLOSED")

    def process_query(self, prompt: str) -> Dict[str, Any]:
        """Complete cognitive processing with active learning"""
        self.interaction_count += 1

        # Step 1: Retrieve relevant memories
        relevant_memories = self.memory.retrieve_relevant_memories(prompt)

        # Step 2: Generate learning context
        learning_context = self._build_learning_context(prompt, relevant_memories)

        # Step 3: Call LLM with enhanced context
        llm_response = self._call_llm_with_learning(learning_context)

        # Step 4: Render to VRAM
        self._render_to_vram(prompt, llm_response)

        # Step 5: Calculate complexity metrics
        complexity = self.display.calculate_complexity()
        vram_hash = self.display.get_vram_hash()
        vram_state = self.display.compress_vram_state()

        # Step 6: Save to memory
        pattern_type = self._classify_pattern(prompt, llm_response)
        interaction_id = self.memory.save_interaction(
            prompt, llm_response, vram_state, vram_hash, complexity, pattern_type
        )

        # Step 7: Active learning analysis
        self._perform_active_learning()

        # Step 8: Check for system instability
        if self.display.detect_instability():
            self._handle_instability()

        return {
            'interaction_id': interaction_id,
            'response': llm_response,
            'complexity': complexity,
            'relevant_memories_used': len(relevant_memories),
            'learning_progress': self.learning_progress,
            'system_stable': not self.display.detect_instability()
        }

    def _build_learning_context(self, prompt: str, memories: List[Dict]) -> str:
        """Build learning-enhanced context for LLM"""
        context = "LEARNING FROM PAST EXPERIENCES:\n\n"

        if memories:
            context += "RELEVANT PAST SUCCESSES:\n"
            for i, memory in enumerate(memories, 1):
                context += f"{i}. Previous request: {memory['prompt']}\n"
                context += f"   Response: {memory['response'][:100]}...\n"
                context += f"   Quality: {memory['quality_score']:.2f}\n\n"
        else:
            context += "No directly relevant past experiences found.\n\n"

        # Add recent insights
        if self.recent_insights:
            context += "RECENT LEARNING INSIGHTS:\n"
            for insight in self.recent_insights[-2:]:  # Last 2 insights
                context += f"- {insight['content']}\n"
            context += "\n"

        context += f"CURRENT REQUEST: {prompt}\n\n"
        context += "INSTRUCTIONS: Build upon successful past approaches. "
        context += "Avoid patterns that had low quality scores. "
        context += "Explain how your solution improves upon previous attempts."

        return context

    def _call_llm_with_learning(self, context: str) -> str:
        """Call LM Studio with learning-enhanced context"""
        try:
            response = requests.post(
                f"{self.lmstudio_host}/v1/chat/completions",
                json={
                    "model": "gemini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a self-improving AI substrate. You learn from your past successes and failures. Your responses should demonstrate continuous improvement and build upon proven patterns."
                        },
                        {"role": "user", "content": context}
                    ],
                    "max_tokens": 500,
                    "temperature": 0.3  # Lower temp for consistent improvement
                },
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: LLM unavailable (status {response.status_code})"

        except Exception as e:
            return f"Connection failed: {str(e)}"

    def _render_to_vram(self, prompt: str, response: str):
        """Render interaction to VRAM display"""
        # Clear display
        self.display.clear_region(0, 0, self.display.width, self.display.height)

        # Render conversation
        self.display.render_text(f"USER: {prompt}", 20, 20, (0, 255, 0))
        self.display.render_text(f"SUBSTRATE: {response}", 20, 50, (0, 255, 255))

        # Render learning status
        status = f"Learning Progress: {self.learning_progress:.1%} | Interactions: {self.interaction_count}"
        self.display.render_text(status, 20, 100, (255, 255, 0))

    def _classify_pattern(self, prompt: str, response: str) -> str:
        """Classify interaction pattern type"""
        text = f"{prompt} {response}".lower()

        if any(word in text for word in ['clock', 'oscillat', 'timer']):
            return "clock_circuit"
        elif any(word in text for word in ['memory', 'store', 'register']):
            return "memory_cell"
        elif any(word in text for word in ['wire', 'connect', 'network']):
            return "wire_network"
        elif any(word in text for word in ['add', 'calculate', 'compute']):
            return "arithmetic"
        else:
            return "generic_pattern"

    def _perform_active_learning(self):
        """Perform active learning analysis"""
        # Get recent interactions for analysis
        recent = self._get_recent_interactions(10)

        if len(recent) >= 3:
            # Generate new insight
            insight = self.memory.generate_learning_insight(recent)
            if insight:
                self.recent_insights.append(insight)
                print(f"💡 NEW INSIGHT: {insight['content']}")

            # Update learning progress
            self._update_learning_progress(recent)

    def _get_recent_interactions(self, count: int) -> List[Dict]:
        """Get recent interactions from memory"""
        return self.memory.get_recent_interactions(count)

    def _update_learning_progress(self, recent_interactions: List[Dict]):
        """Update overall learning progress"""
        if not recent_interactions:
            return

        # Calculate average quality of recent interactions
        avg_quality = sum(i.get('quality_score', 0) for i in recent_interactions) / len(recent_interactions)

        # Progress based on quality and diversity
        quality_component = avg_quality * 0.7
        diversity_component = min(len(recent_interactions) / 10.0, 0.3)

        self.learning_progress = quality_component + diversity_component

    def _handle_instability(self):
        """Handle system instability detected by cellular automata metrics"""
        print("🚨 SYSTEM INSTABILITY DETECTED - Performing self-correction...")

        # Reset to stable state
        self.display.clear_region(0, 0, self.display.width, self.display.height)
        self.display.render_text("SYSTEM RECOVERY: Learning reset", 20, 20, (255, 0, 0))

        # Log recovery insight
        self.memory._save_insight({
            'type': 'system_recovery',
            'content': 'System recovered from low-complexity state via hard reset',
            'confidence': 0.9
        })

    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        return {
            'interaction_count': self.interaction_count,
            'learning_progress': self.learning_progress,
            'current_complexity': self.display.calculate_complexity(),
            'system_stable': not self.display.detect_instability(),
            'recent_insights': len(self.recent_insights),
            'vram_hash': self.display.get_vram_hash()
        }
