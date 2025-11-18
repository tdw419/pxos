"""
pxIR Optimizer - Constant Folding + Dead Code Elimination (v0.1)

Pipeline:

    Python → pxIR → [optimizer.py] → optimized pxIR → PXI → Primitives → Binary

This file implements:
- ConstantFolding: fold simple arithmetic on literals
- DeadCodeElimination: remove unused, side-effect-free ops
- PassManager: run multiple passes to a fixed point

Inspired by:
- LLVM's pass infrastructure
- GCC's optimization pipeline
- TVM's relay optimizer
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set

from .ir import Program, Block, Op, Value, TypeKind


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _is_const_value(v: Value) -> bool:
    """Return True if this Value represents a compile-time constant."""
    return v.is_const


def _get_const(v: Value):
    """Return Python value for a constant Value."""
    return v.const_value


def _make_const_like(orig: Value, new_val, builder=None) -> Value:
    """Create a new constant Value with type = orig.type and const_value=new_val."""
    # Generate a unique name for the folded constant
    import random
    const_name = f"%const_{abs(hash(new_val)) % 10000}"
    return Value(
        name=const_name,
        ty=orig.ty,
        is_const=True,
        const_value=new_val,
    )


# -----------------------------------------------------------------------------
# Constant Folding
# -----------------------------------------------------------------------------

@dataclass
class ConstantFolding:
    """
    Fold simple arithmetic and comparison ops on constant operands.

    Handles ops like:
        ADD, SUB, MUL, DIV, MOD
        RELU (on scalar)

    Examples:
        %1 = ADD(5, 3) → %1 = 8
        %2 = MUL(2, 4) → %2 = 8
        %3 = RELU(-5) → %3 = 0
    """

    changes_made: int = 0

    def run(self, prog: Program) -> bool:
        self.changes_made = 0

        for block in prog.blocks:
            for op in block.ops:
                # Skip ops without result or non-arithmetic ops
                if op.result is None:
                    continue

                # Only fold if ALL operands are constant Values
                if not op.operands:
                    continue
                if not all(isinstance(o, Value) and _is_const_value(o) for o in op.operands):
                    continue

                # Extract Python constants
                consts = [_get_const(o) for o in op.operands]
                if any(c is None for c in consts):
                    continue

                new_val = None

                # Arithmetic operations
                if op.op == "ADD" and len(consts) == 2:
                    new_val = consts[0] + consts[1]
                elif op.op == "SUB" and len(consts) == 2:
                    new_val = consts[0] - consts[1]
                elif op.op == "MUL" and len(consts) == 2:
                    new_val = consts[0] * consts[1]
                elif op.op == "DIV" and len(consts) == 2:
                    # Simple integer division (avoid div-by-zero)
                    if consts[1] != 0:
                        if isinstance(consts[0], float) or isinstance(consts[1], float):
                            new_val = consts[0] / consts[1]
                        else:
                            new_val = consts[0] // consts[1]
                elif op.op == "MOD" and len(consts) == 2:
                    if consts[1] != 0:
                        new_val = consts[0] % consts[1]
                elif op.op == "RELU" and len(consts) == 1:
                    c = consts[0]
                    # Works for scalar ints/floats
                    new_val = c if c > 0 else 0

                # Algebraic simplifications (bonus!)
                elif op.op == "ADD" and len(op.operands) == 2:
                    # x + 0 = x
                    if isinstance(op.operands[1], Value) and _is_const_value(op.operands[1]):
                        if _get_const(op.operands[1]) == 0:
                            prog.replace_all_uses(op.result, op.operands[0])
                            self.changes_made += 1
                            continue
                elif op.op == "MUL" and len(op.operands) == 2:
                    # x * 1 = x
                    if isinstance(op.operands[1], Value) and _is_const_value(op.operands[1]):
                        if _get_const(op.operands[1]) == 1:
                            prog.replace_all_uses(op.result, op.operands[0])
                            self.changes_made += 1
                            continue
                    # x * 0 = 0
                    if isinstance(op.operands[1], Value) and _is_const_value(op.operands[1]):
                        if _get_const(op.operands[1]) == 0:
                            const_v = _make_const_like(op.result, 0)
                            prog.replace_all_uses(op.result, const_v)
                            self.changes_made += 1
                            continue

                if new_val is None:
                    continue  # nothing to fold

                # Replace the result Value with a new constant Value
                const_v = _make_const_like(op.result, new_val)

                # Replace all uses of op.result with const_v
                prog.replace_all_uses(op.result, const_v)

                # This op is now dead if it has no side effects; DCE will remove it
                self.changes_made += 1

        return self.changes_made > 0


# -----------------------------------------------------------------------------
# Dead Code Elimination
# -----------------------------------------------------------------------------

@dataclass
class DeadCodeElimination:
    """
    Remove ops whose results are never used and which have no side effects.

    We keep:
      - Ops with side effects (e.g. PRINT_STR, DRAW_GLYPH, STORE, SYSCALL)
      - Ops whose result is used by other ops or is a function return

    Examples:
        %1 = ADD(%x, %y)
        # %1 is never used
        → DELETE this op

        PRINT_STR("hello")
        # Side effect, keep even though no result
        → KEEP this op
    """

    changes_made: int = 0

    def run(self, prog: Program) -> bool:
        self.changes_made = 0

        # Compute used values: any operand or terminator input counts as a use
        used: Set[Value] = set()

        for block in prog.blocks:
            for op in block.ops:
                for o in op.operands:
                    if isinstance(o, Value):
                        used.add(o)

        # Also treat any function returns as "used"
        for ret in prog.get_return_values():
            if isinstance(ret, Value):
                used.add(ret)

        # Now remove ops whose result is unused and which have no side effects
        for block in prog.blocks:
            new_ops: List[Op] = []
            for op in block.ops:
                if op.result is None:
                    # e.g. store, print, branch – keep them (side effects or control)
                    new_ops.append(op)
                    continue

                if op.has_side_effects:
                    # Must keep even if result unused
                    new_ops.append(op)
                    continue

                if op.result not in used:
                    # Truly dead
                    self.changes_made += 1
                    continue  # drop op

                new_ops.append(op)

            block.ops = new_ops

        return self.changes_made > 0


# -----------------------------------------------------------------------------
# Common Subexpression Elimination (bonus!)
# -----------------------------------------------------------------------------

@dataclass
class CommonSubexpressionElimination:
    """
    Remove redundant computations by reusing previously computed values.

    Example:
        %1 = ADD(%a, %b)
        %2 = MUL(%1, %c)
        %3 = ADD(%a, %b)  # Same as %1!
        → Replace %3 with %1
    """

    changes_made: int = 0

    def run(self, prog: Program) -> bool:
        self.changes_made = 0

        for block in prog.blocks:
            # Map from (op, operands_tuple) to result Value
            computed: dict = {}

            new_ops: List[Op] = []
            for op in block.ops:
                # Skip side-effecting ops
                if op.has_side_effects or op.result is None:
                    new_ops.append(op)
                    continue

                # Create a key for this computation
                operands_tuple = tuple(
                    o.name if isinstance(o, Value) else o
                    for o in op.operands
                )
                key = (op.op, operands_tuple)

                if key in computed:
                    # Reuse previous computation
                    prev_result = computed[key]
                    prog.replace_all_uses(op.result, prev_result)
                    self.changes_made += 1
                    # Don't add this op
                else:
                    # First time seeing this computation
                    computed[key] = op.result
                    new_ops.append(op)

            block.ops = new_ops

        return self.changes_made > 0


# -----------------------------------------------------------------------------
# Pass Manager
# -----------------------------------------------------------------------------

@dataclass
class PassManager:
    """
    Run passes repeatedly until no changes occur or max_iters reached.

    This is the GCC-style fixed-point iteration approach.
    """

    passes: List[object]
    max_iters: int = 5
    verbose: bool = False

    def run(self, prog: Program) -> None:
        """Run all passes until fixed point."""
        for iteration in range(self.max_iters):
            if self.verbose:
                print(f"\n=== Pass iteration {iteration + 1} ===")

            any_change = False
            for p in self.passes:
                pass_name = p.__class__.__name__
                if self.verbose:
                    print(f"Running {pass_name}...", end=" ")

                changed = p.run(prog)
                if changed:
                    any_change = True
                    if self.verbose:
                        changes = getattr(p, 'changes_made', '?')
                        print(f"✓ ({changes} changes)")
                else:
                    if self.verbose:
                        print("(no changes)")

            if not any_change:
                if self.verbose:
                    print(f"\n✓ Converged after {iteration + 1} iteration(s)")
                break


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------

def optimize_program(prog: Program, level: int = 2, verbose: bool = False) -> Program:
    """
    Apply a standard set of pxIR optimizations.

    Optimization levels:
      -O0: No optimization
      -O1: Basic (constant folding, DCE)
      -O2: + CSE
      -O3: + aggressive passes (future)

    Currently:
      - ConstantFolding
      - DeadCodeElimination
      - CommonSubexpressionElimination (O2+)
    """
    if level == 0:
        return prog

    passes = [
        ConstantFolding(),
        DeadCodeElimination(),
    ]

    if level >= 2:
        passes.insert(1, CommonSubexpressionElimination())

    pm = PassManager(passes=passes, verbose=verbose)
    pm.run(prog)
    return prog
