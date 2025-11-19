#!/usr/bin/env python3
"""
LLM Primitive Analyzer
Uses Language Models to analyze primitives and make intelligent decisions

Supports:
- OpenAI API (GPT-4)
- Anthropic API (Claude)
- Local models via llama.cpp or similar

This is the KEY INNOVATION: LLM understands intent and optimizes execution strategy
"""

import json
import os
from typing import Dict, List, Optional, Literal
from dataclasses import dataclass
from enum import Enum


class ExecutionTarget(Enum):
    """Where should code execute?"""
    CPU = "cpu"
    GPU = "gpu"
    HYBRID = "hybrid"  # Split between CPU and GPU
    AUTO = "auto"      # LLM decides


@dataclass
class AnalysisResult:
    """Result of LLM primitive analysis"""
    target: ExecutionTarget
    reasoning: str
    estimated_speedup: Optional[float] = None
    warnings: List[str] = None
    suggested_optimizations: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.suggested_optimizations is None:
            self.suggested_optimizations = []


class LLMAnalyzer:
    """Analyzes primitives using LLM intelligence"""

    def __init__(self, provider: Literal["openai", "anthropic", "local"] = "local"):
        self.provider = provider
        self._setup_client()

    def _setup_client(self):
        """Setup LLM client based on provider"""
        if self.provider == "openai":
            try:
                import openai
                self.client = openai
                self.api_key = os.getenv("OPENAI_API_KEY")
                if not self.api_key:
                    print("Warning: OPENAI_API_KEY not set")
            except ImportError:
                print("Warning: openai package not installed")
                self.client = None

        elif self.provider == "anthropic":
            try:
                import anthropic
                self.client = anthropic.Anthropic(
                    api_key=os.getenv("ANTHROPIC_API_KEY")
                )
            except ImportError:
                print("Warning: anthropic package not installed")
                self.client = None

        else:  # local
            self.client = None
            print("Using rule-based analysis (no LLM)")

    def analyze_primitive(self, primitive_code: str, context: Dict = None) -> AnalysisResult:
        """
        Analyze a high-level primitive and decide execution strategy

        Args:
            primitive_code: The primitive code to analyze
            context: Optional context (hardware info, data size, etc.)

        Returns:
            AnalysisResult with target and reasoning
        """

        if self.client and self.provider == "openai":
            return self._analyze_with_openai(primitive_code, context)
        elif self.client and self.provider == "anthropic":
            return self._analyze_with_anthropic(primitive_code, context)
        else:
            return self._analyze_with_rules(primitive_code, context)

    def _analyze_with_openai(self, primitive_code: str, context: Dict) -> AnalysisResult:
        """Analyze using OpenAI GPT-4"""

        prompt = self._build_analysis_prompt(primitive_code, context)

        try:
            response = self.client.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in heterogeneous computing and GPU optimization. "
                                 "Analyze code primitives and decide if they should run on CPU or GPU."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent decisions
            )

            result_text = response.choices[0].message.content
            return self._parse_llm_response(result_text)

        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._analyze_with_rules(primitive_code, context)

    def _analyze_with_anthropic(self, primitive_code: str, context: Dict) -> AnalysisResult:
        """Analyze using Anthropic Claude"""

        prompt = self._build_analysis_prompt(primitive_code, context)

        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                system="You are an expert in heterogeneous computing and GPU optimization."
            )

            result_text = message.content[0].text
            return self._parse_llm_response(result_text)

        except Exception as e:
            print(f"Anthropic API error: {e}")
            return self._analyze_with_rules(primitive_code, context)

    def _analyze_with_rules(self, primitive_code: str, context: Dict) -> AnalysisResult:
        """Rule-based analysis (fallback when no LLM available)"""

        code_lower = primitive_code.lower()

        # Rule 1: Parallel operations → GPU
        parallel_keywords = ['parallel_for', 'parallel_sort', 'parallel_reduce', 'gpu_kernel']
        if any(kw in code_lower for kw in parallel_keywords):
            return AnalysisResult(
                target=ExecutionTarget.GPU,
                reasoning="Contains parallel operations - suitable for GPU",
                estimated_speedup=10.0,
                suggested_optimizations=["Ensure data is contiguous in memory"]
            )

        # Rule 2: I/O operations → CPU
        io_keywords = ['read', 'write', 'open', 'close', 'print', 'scanf']
        if any(kw in code_lower for kw in io_keywords):
            return AnalysisResult(
                target=ExecutionTarget.CPU,
                reasoning="Contains I/O operations - must run on CPU",
                warnings=["GPU cannot perform I/O operations"]
            )

        # Rule 3: Small datasets → CPU
        if context and context.get('data_size'):
            size = context['data_size']
            if size < 1000:
                return AnalysisResult(
                    target=ExecutionTarget.CPU,
                    reasoning=f"Dataset too small ({size} elements) - GPU overhead not worth it",
                    warnings=["GPU launch overhead dominates for small data"]
                )

        # Rule 4: Large matrix operations → GPU
        matrix_keywords = ['matrix', 'matmul', 'gemm', 'dot product']
        if any(kw in code_lower for kw in matrix_keywords):
            return AnalysisResult(
                target=ExecutionTarget.GPU,
                reasoning="Matrix operations highly parallel - excellent for GPU",
                estimated_speedup=50.0,
                suggested_optimizations=["Use cuBLAS for optimal performance"]
            )

        # Default: CPU
        return AnalysisResult(
            target=ExecutionTarget.CPU,
            reasoning="No clear parallelism detected - defaulting to CPU"
        )

    def _build_analysis_prompt(self, primitive_code: str, context: Dict) -> str:
        """Build prompt for LLM analysis"""

        prompt = f"""Analyze this code primitive and decide if it should run on CPU or GPU:

```
{primitive_code}
```

Context:
"""
        if context:
            for key, value in context.items():
                prompt += f"- {key}: {value}\n"

        prompt += """
Provide your analysis in this format:

TARGET: [CPU or GPU or HYBRID]
REASONING: [Explain why]
SPEEDUP: [Estimated speedup if using GPU, or "N/A"]
WARNINGS: [Any warnings or concerns]
OPTIMIZATIONS: [Suggested optimizations]

Consider:
1. Is the operation data-parallel?
2. How large is the dataset?
3. Does it involve I/O or branching?
4. What's the GPU launch overhead vs computation time?
5. Memory transfer costs

Be specific and practical.
"""
        return prompt

    def _parse_llm_response(self, response_text: str) -> AnalysisResult:
        """Parse LLM response into AnalysisResult"""

        lines = response_text.strip().split('\n')
        target = ExecutionTarget.CPU
        reasoning = ""
        speedup = None
        warnings = []
        optimizations = []

        for line in lines:
            line = line.strip()

            if line.startswith("TARGET:"):
                target_str = line.split(":", 1)[1].strip().upper()
                if "GPU" in target_str:
                    target = ExecutionTarget.GPU
                elif "HYBRID" in target_str:
                    target = ExecutionTarget.HYBRID

            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

            elif line.startswith("SPEEDUP:"):
                speedup_str = line.split(":", 1)[1].strip()
                try:
                    speedup = float(speedup_str.replace('x', '').replace('N/A', '0'))
                    if speedup == 0:
                        speedup = None
                except:
                    speedup = None

            elif line.startswith("WARNINGS:"):
                warn_text = line.split(":", 1)[1].strip()
                if warn_text and warn_text != "None":
                    warnings.append(warn_text)

            elif line.startswith("OPTIMIZATIONS:"):
                opt_text = line.split(":", 1)[1].strip()
                if opt_text and opt_text != "None":
                    optimizations.append(opt_text)

        return AnalysisResult(
            target=target,
            reasoning=reasoning,
            estimated_speedup=speedup,
            warnings=warnings,
            suggested_optimizations=optimizations
        )


def main():
    """Test the LLM analyzer"""

    analyzer = LLMAnalyzer(provider="local")  # Use rule-based for demo

    # Test case 1: Parallel sort (should be GPU)
    test1 = """
    PARALLEL_SORT array SIZE 1000000 ASCENDING
    """

    result1 = analyzer.analyze_primitive(test1, {"data_size": 1000000})
    print("Test 1: Parallel Sort")
    print(f"  Target: {result1.target.value}")
    print(f"  Reasoning: {result1.reasoning}")
    print(f"  Speedup: {result1.estimated_speedup}x")
    print()

    # Test case 2: File I/O (should be CPU)
    test2 = """
    OPEN file "/data/input.txt"
    READ file INTO buffer
    """

    result2 = analyzer.analyze_primitive(test2)
    print("Test 2: File I/O")
    print(f"  Target: {result2.target.value}")
    print(f"  Reasoning: {result2.reasoning}")
    print(f"  Warnings: {result2.warnings}")
    print()

    # Test case 3: Small loop (should be CPU)
    test3 = """
    FOR i FROM 0 TO 10:
        PRINT i
    """

    result3 = analyzer.analyze_primitive(test3, {"data_size": 10})
    print("Test 3: Small Loop")
    print(f"  Target: {result3.target.value}")
    print(f"  Reasoning: {result3.reasoning}")
    print()


if __name__ == "__main__":
    main()
