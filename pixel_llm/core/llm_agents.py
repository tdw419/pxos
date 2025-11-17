#!/usr/bin/env python3
"""
LLM Agent Integration for Pixel-LLM Coaching

Provides interfaces to:
1. Gemini (via API or gemini-cli) - For reviews and coaching
2. Local LLM (via llama.cpp or ollama) - For code generation

The coaching pattern:
    Local LLM generates → Gemini reviews → Iterate until good
"""

import os
import json
import subprocess
import re
from typing import Tuple, Optional, Dict
from pathlib import Path


class GeminiAgent:
    """
    Interface to Gemini for code reviews and coaching.

    Uses gemini-cli if available, falls back to direct API.
    """

    def __init__(self):
        self.has_cli = self._check_gemini_cli()
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.api_available = self._check_api_available() if self.api_key else False

        if not self.has_cli and not self.api_key:
            print("⚠️  Warning: No Gemini access found!")
            print("   Set GEMINI_API_KEY or install gemini-cli")
        elif self.api_key and not self.api_available:
            print("⚠️  Warning: GEMINI_API_KEY set but API not accessible")

    def _check_api_available(self) -> bool:
        """Check if Gemini API is accessible"""
        if not self.api_key:
            return False

        # Quick check - just verify requests library is available
        try:
            import requests
            return True
        except ImportError:
            print("ℹ️  Install 'requests' library for Gemini API access")
            return False

    def get_capabilities(self) -> Dict[str, any]:
        """Get detailed capability information"""
        return {
            "has_cli": self.has_cli,
            "has_api_key": self.api_key is not None,
            "api_available": self.api_available,
            "ready": self.has_cli or (self.api_key and self.api_available),
            "method": "cli" if self.has_cli else ("api" if self.api_available else "none")
        }

    def _check_gemini_cli(self) -> bool:
        """Check if gemini-cli is available"""
        try:
            result = subprocess.run(
                ["gemini-cli", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def review_code(
        self,
        code: str,
        task: Dict,
        iteration: int = 1
    ) -> Tuple[int, str]:
        """
        Review code for a pixel-LLM task.

        Args:
            code: Generated code to review
            task: Task specification
            iteration: Which iteration (1, 2, 3)

        Returns:
            (score, feedback) where score is 1-10
        """
        title = task.get('title', 'Unknown')
        phase = task.get('phase', 'unknown')
        path = task.get('path', '')
        description = task.get('description', '')

        # Build focused review prompt
        prompt = f"""You are reviewing code for the Pixel-LLM project - an AI that lives IN pixels and runs on GPU.

**Task**: {title}
**Phase**: {phase}
**File**: {path}
**Iteration**: {iteration}/3

**Goal**:
{description[:300]}

**Code to Review** ({len(code)} chars):
```python
{code[:2500]}
{"..." if len(code) > 2500 else ""}
```

**Review Criteria**:
1. Pixel/Spatial Concepts: Does it handle pixel operations correctly?
2. GPU Integration: Will it work with WGSL/PixelFS/InfiniteMap?
3. Production Quality: Error handling, documentation, edge cases?
4. Pixel-Native Vision: Does it advance substrate-native intelligence?
5. Code Completeness: Is it a stub or full implementation?

**Important**:
- Score HARSHLY. Stubs/incomplete code = 1-3
- Well-structured but needs work = 4-6
- Production ready = 7-9
- Exceptional = 10

**Output Format**:
SCORE: [1-10]

FEEDBACK:
[Specific improvements needed. Be constructive but demanding.]

PIXEL-SPECIFIC NOTES:
[Any unique considerations for pixel-native AI]
"""

        # Call Gemini
        response = self._call_gemini(prompt)

        if not response:
            print("⚠️  Gemini review failed, defaulting to score 5")
            return 5, "Could not get Gemini review"

        # Parse response
        score, feedback = self._parse_review(response)

        return score, feedback

    def _call_gemini(self, prompt: str) -> Optional[str]:
        """Call Gemini via CLI or API"""

        if self.has_cli:
            return self._call_gemini_cli(prompt)
        elif self.api_key:
            return self._call_gemini_api(prompt)
        else:
            return None

    def _call_gemini_cli(self, prompt: str) -> Optional[str]:
        """Call via gemini-cli"""
        try:
            result = subprocess.run(
                ["gemini-cli", "--yes", "--message", prompt],
                capture_output=True,
                text=True,
                timeout=60
            )

            if result.returncode == 0:
                return result.stdout.strip()
            else:
                print(f"gemini-cli error: {result.stderr}")
                return None

        except subprocess.TimeoutExpired:
            print("Gemini CLI timeout")
            return None
        except Exception as e:
            print(f"Gemini CLI error: {e}")
            return None

    def _call_gemini_api(self, prompt: str) -> Optional[str]:
        """Call via direct API (requires requests library)"""
        try:
            import requests

            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

            headers = {
                "Content-Type": "application/json"
            }

            data = {
                "contents": [{
                    "parts": [{
                        "text": prompt
                    }]
                }]
            }

            response = requests.post(
                f"{url}?key={self.api_key}",
                headers=headers,
                json=data,
                timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result['candidates'][0]['content']['parts'][0]['text']
            else:
                print(f"API error: {response.status_code}")
                return None

        except ImportError:
            print("requests library not installed for API calls")
            return None
        except Exception as e:
            print(f"API error: {e}")
            return None

    def _parse_review(self, response: str) -> Tuple[int, str]:
        """Parse Gemini's review response"""

        # Extract score
        score_match = re.search(r'SCORE:\s*(\d+)', response)
        score = int(score_match.group(1)) if score_match else 5

        # Clamp to 1-10
        score = max(1, min(10, score))

        # Extract feedback
        feedback_match = re.search(r'FEEDBACK:\s*(.+?)(?=PIXEL-SPECIFIC|$)', response, re.DOTALL)
        feedback = feedback_match.group(1).strip() if feedback_match else response

        return score, feedback


class LocalLLMAgent:
    """
    Interface to local LLM for code generation.

    Supports:
    - llama.cpp via llama-cli
    - Ollama
    - Any subprocess-based LLM
    """

    def __init__(self):
        self.backend = self._detect_backend()
        self.model_available = False

        if self.backend:
            self.model_available = self._check_model_available()

        if not self.backend:
            print("⚠️  Warning: No local LLM found!")
            print("   Install llama.cpp or ollama")
        elif not self.model_available:
            print("⚠️  Warning: LLM backend found but model not available")

    def _check_model_available(self) -> bool:
        """Check if the required model is available"""
        if self.backend == "ollama":
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                return "qwen2.5-coder" in result.stdout
            except:
                return False
        # For llama-cli and llama-cpp-python, assume model is available
        # (user is responsible for model path)
        return True

    def get_capabilities(self) -> Dict[str, any]:
        """Get detailed capability information"""
        caps = {
            "backend": self.backend,
            "model_available": self.model_available,
            "ready": self.backend is not None and self.model_available,
        }

        if self.backend == "ollama":
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # Parse available models
                models = []
                for line in result.stdout.split('\n')[1:]:  # Skip header
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
                caps["available_models"] = models
            except:
                caps["available_models"] = []

        return caps

    def _detect_backend(self) -> Optional[str]:
        """Detect which LLM backend is available"""

        # Check ollama (preferred for ease of use)
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Verify qwen2.5-coder model exists
                if "qwen2.5-coder" in result.stdout:
                    return "ollama"
                # Ollama exists but model not pulled
                print("ℹ️  Ollama found but qwen2.5-coder:7b not installed")
                print("   Run: ollama pull qwen2.5-coder:7b")
                # Still return ollama as backend exists
                return "ollama"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check llama.cpp
        try:
            result = subprocess.run(
                ["llama-cli", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return "llama-cli"
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Check for llama-cpp-python (Python bindings)
        try:
            import llama_cpp
            # If import succeeds, we can use Python bindings
            return "llama-cpp-python"
        except ImportError:
            pass

        return None

    def generate_code(
        self,
        task: Dict,
        feedback: Optional[str] = None,
        previous_code: Optional[str] = None
    ) -> str:
        """
        Generate code for a task.

        Args:
            task: Task specification
            feedback: Feedback from previous iteration
            previous_code: Previous attempt (if iteration > 1)

        Returns:
            Generated code
        """
        title = task.get('title', 'Unknown')
        description = task.get('description', '')
        path = task.get('path', '')
        phase = task.get('phase', 'unknown')

        # Build generation prompt
        if feedback and previous_code:
            # Iteration 2+
            prompt = f"""You are implementing code for Pixel-LLM - an AI that lives in pixels.

**Previous Attempt** had issues. Improve it based on feedback.

**Task**: {title}
**File**: {path}
**Phase**: {phase}

**Previous Code**:
```python
{previous_code[:1500]}
```

**Feedback from Review**:
{feedback}

**Instructions**:
1. Address ALL feedback points
2. Maintain pixel/spatial focus
3. Add comprehensive docstrings
4. Include error handling
5. Make it production-ready

**Generate ONLY the complete, improved Python code. No explanations.**
"""
        else:
            # First iteration
            prompt = f"""You are implementing code for Pixel-LLM - an AI that lives in pixels and runs on GPU.

**Task**: {title}
**File**: {path}
**Phase**: {phase}

**Requirements**:
{description}

**Context**:
- PixelFS stores data as RGB pixels (see pixel_llm/core/pixelfs.py)
- InfiniteMap manages 2D spatial memory (see pixel_llm/core/infinite_map.py)
- This code should integrate with these systems

**Instructions**:
1. Write complete, production-ready code
2. Include comprehensive docstrings
3. Add error handling
4. Focus on pixel/spatial operations
5. Make it 400+ lines with examples/tests

**Generate ONLY the complete Python code. No explanations, just code.**
"""

        # Call local LLM
        code = self._call_local_llm(prompt)

        return code or self._generate_stub(task)

    def _call_local_llm(self, prompt: str) -> Optional[str]:
        """Call local LLM backend"""

        if self.backend == "llama-cli":
            return self._call_llama_cli(prompt)
        elif self.backend == "ollama":
            return self._call_ollama(prompt)
        else:
            return None

    def _call_llama_cli(self, prompt: str) -> Optional[str]:
        """Call llama.cpp"""
        try:
            # Assuming model is already loaded
            result = subprocess.run(
                [
                    "llama-cli",
                    "-m", "models/qwen2.5-7b-instruct.gguf",  # Adjust path
                    "-p", prompt,
                    "-n", "2048",
                    "--temp", "0.7",
                ],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                # Extract code from response
                return self._extract_code(result.stdout)
            else:
                return None

        except Exception as e:
            print(f"llama-cli error: {e}")
            return None

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Call ollama"""
        try:
            result = subprocess.run(
                ["ollama", "run", "qwen2.5-coder:7b", prompt],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                return self._extract_code(result.stdout)
            else:
                return None

        except Exception as e:
            print(f"ollama error: {e}")
            return None

    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response"""

        # Look for code blocks
        code_match = re.search(r'```python\n(.+?)\n```', response, re.DOTALL)
        if code_match:
            return code_match.group(1)

        # Fallback: return whole response
        return response.strip()

    def _generate_stub(self, task: Dict) -> str:
        """Generate a basic stub if LLM fails"""
        title = task.get('title', 'Unknown')
        path = task.get('path', 'unknown.py')

        return f'''#!/usr/bin/env python3
"""
{title}

TODO: Implement this component for Pixel-LLM
"""

def main():
    """Main entry point"""
    print("Stub implementation for: {title}")
    pass

if __name__ == "__main__":
    main()
'''


# Test/demo
if __name__ == "__main__":
    print("\n" + "="*60)
    print("LLM AGENT CAPABILITIES")
    print("="*60)

    # Test Gemini
    print("\n🔍 Gemini Agent:")
    gemini = GeminiAgent()
    gemini_caps = gemini.get_capabilities()

    for key, value in gemini_caps.items():
        icon = "✅" if value and key != "method" else "❌"
        if key == "method":
            icon = "📡"
        print(f"  {icon} {key}: {value}")

    # Test Local LLM
    print("\n🔍 Local LLM Agent:")
    local = LocalLLMAgent()
    local_caps = local.get_capabilities()

    for key, value in local_caps.items():
        if key == "available_models":
            print(f"  📦 {key}:")
            for model in value:
                print(f"     - {model}")
        else:
            icon = "✅" if value else "❌"
            print(f"  {icon} {key}: {value}")

    # Summary
    print("\n" + "="*60)
    if gemini_caps["ready"] and local_caps["ready"]:
        print("✅ COACHING SYSTEM READY")
        print("   Gemini will review, Local LLM will generate")
    elif local_caps["ready"]:
        print("⚠️  LOCAL LLM ONLY")
        print("   Can generate but no Gemini review")
    elif gemini_caps["ready"]:
        print("⚠️  GEMINI ONLY")
        print("   Can review but no local generation")
    else:
        print("❌ NO LLM AGENTS CONFIGURED")
        print("\n   Quick setup:")
        print("   1. Install ollama: curl -fsSL https://ollama.ai/install.sh | sh")
        print("   2. Pull model: ollama pull qwen2.5-coder:7b")
        print("   3. Set Gemini key: export GEMINI_API_KEY='your-key'")
    print("="*60 + "\n")
