"""
Learning-Enhanced LLM
Uses saved interactions to improve responses over time
"""

from vram_learning import VRAMLearningSystem
import requests
import json
import numpy as np
from typing import Dict, Any, List

class LearningEnhancedLLM:
    def __init__(self, substrate, lmstudio_host="http://localhost:1234"):
        self.substrate = substrate
        self.lmstudio_host = lmstudio_host
        self.learning_system = VRAMLearningSystem(substrate)

        print("🎓 Learning-Enhanced LLM Initialized")
        print("   LLM will improve using saved interactions")

    def learn_and_respond(self, prompt: str, context_region: tuple = None) -> Dict[str, Any]:
        """
        Enhanced LLM response that uses learning from previous interactions
        """
        # 1. Find similar previous interactions
        similar_patterns = self._get_relevant_learning(prompt)

        # 2. Build learning-enhanced prompt
        enhanced_prompt = self._build_learning_prompt(prompt, similar_patterns)

        # 3. Call LLM with learning context
        response = self._call_learning_llm(enhanced_prompt)

        # 4. Extract and execute pattern
        execution_result = self._execute_with_learning(response, similar_patterns)

        # 5. Save interaction for future learning
        if execution_result["success"]:
            pattern_hash = self.learning_system.save_interaction(
                prompt, response,
                execution_result["coordinates"],
                execution_result["pattern"]
            )

            # Record success with efficiency metrics
            self.learning_system.record_success(
                pattern_hash,
                execution_result.get("efficiency_metrics", {})
            )

        return {
            "response": response,
            "execution": execution_result,
            "learning_used": len(similar_patterns),
            "pattern_hash": pattern_hash if execution_result["success"] else None
        }

    def _get_relevant_learning(self, prompt: str) -> List[Dict]:
        """Get relevant learning from previous interactions"""
        relevant_patterns = []

        # Search interaction history for similar prompts
        for interaction in self.learning_system.interaction_history[:20]:  # Recent ones
            if self._prompt_similarity(prompt, interaction['prompt']) > 0.5:
                pattern_data = self.learning_system.pattern_library.get(
                    interaction['pattern_hash']
                )
                if pattern_data:
                    relevant_patterns.append({
                        'prompt': interaction['prompt'],
                        'response': interaction['response'],
                        'pattern': pattern_data['pattern'],
                        'efficiency': pattern_data['efficiency_score'],
                        'success_count': pattern_data['success_count']
                    })

        return relevant_patterns

    def _prompt_similarity(self, prompt1: str, prompt2: str) -> float:
        """Calculate similarity between two prompts"""
        words1 = set(prompt1.lower().split())
        words2 = set(prompt2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _build_learning_prompt(self, prompt: str, similar_patterns: List[Dict]) -> str:
        """Build prompt enhanced with learning from previous interactions"""

        learning_context = "PREVIOUS SUCCESSFUL PATTERNS:\n"

        for i, pattern_data in enumerate(similar_patterns[:3]):  # Top 3 most relevant
            learning_context += f"""
Pattern {i+1}:
- Original Request: {pattern_data['prompt']}
- LLM Approach: {pattern_data['response'][:200]}...
- Success Rate: {pattern_data['success_count']} uses
- Efficiency Score: {pattern_data['efficiency']:.2f}
"""

        learning_context += f"""
CURRENT REQUEST: {prompt}

LEARNING OBJECTIVES:
1. Build upon previous successful patterns
2. Avoid approaches that had low efficiency scores
3. Incorporate best practices from high-scoring patterns
4. Explain how this improves upon previous attempts

Respond with both the pixel pattern and learning insights.
"""

        return learning_context

    def _call_learning_llm(self, prompt: str) -> str:
        """Call LLM with learning-enhanced context"""
        try:
            response = requests.post(
                f"{self.lmstudio_host}/v1/chat/completions",
                json={
                    "model": "gemini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a learning pxOS architect. You improve over time by analyzing successful patterns and avoiding inefficient approaches."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 1000,
                    "temperature": 0.6
                }
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"Error: {response.status_code}"

        except Exception as e:
            return f"Connection failed: {str(e)}"

    def _execute_with_learning(self, response: str, similar_patterns: List[Dict]) -> Dict[str, Any]:
        """Execute LLM response with learning optimizations"""
        # This would integrate with your existing pattern execution
        # For now, return a simplified result
        return {
            "success": True,
            "coordinates": (100, 100),
            "pattern": np.zeros((32, 32, 4), dtype=np.float32),  # Placeholder
            "efficiency_metrics": {
                "execution_speed": 0.8,
                "memory_usage": 0.7,
                "pattern_density": 0.6,
                "computational_power": 0.75
            }
        }

    def generate_learning_report(self) -> Dict[str, Any]:
        """Generate a learning progress report"""
        insights = self.learning_system.get_learning_insights()
        top_patterns = sorted(
            self.learning_system.pattern_library.values(),
            key=lambda x: x['efficiency_score'],
            reverse=True
        )[:5]

        return {
            "total_patterns": len(self.learning_system.pattern_library),
            "total_interactions": len(self.learning_system.interaction_history),
            "top_efficient_patterns": [
                {
                    "type": p['type'],
                    "purpose": p['purpose'],
                    "efficiency": p['efficiency_score'],
                    "success_count": p['success_count']
                }
                for p in top_patterns
            ],
            "recent_insights": insights,
            "learning_progress": self._calculate_learning_progress()
        }

    def _calculate_learning_progress(self) -> float:
        """Calculate overall learning progress score"""
        if not self.learning_system.pattern_library:
            return 0.0

        efficiency_scores = [p['efficiency_score'] for p in self.learning_system.pattern_library.values()]
        success_counts = [p['success_count'] for p in self.learning_system.pattern_library.values()]

        avg_efficiency = np.mean(efficiency_scores)
        avg_success = np.mean(success_counts)

        # Combine metrics for overall progress
        progress = (avg_efficiency * 0.7) + (min(avg_success / 10, 1.0) * 0.3)
        return min(progress, 1.0)
