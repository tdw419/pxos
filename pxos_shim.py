#!/usr/bin/env python3
"""
pxOS Launcher - Single Entry Point to Pixel World

This is the ONLY file that lives in the host system's normal filesystem.
Everything else runs from pixel archives.

Usage:
    python pxos_shim.py run <program>           # Run from current cartridge
    python pxos_shim.py run --cartridge <name> <program>  # Run from specific cartridge
    python pxos_shim.py status                  # Show cartridge status
    python pxos_shim.py test --cartridge <name> # Test Genesis compliance
    python pxos_shim.py promote <name>          # Promote cartridge to current
    python pxos_shim.py rollback <name>         # Rollback to previous version

Examples:
    python pxos_shim.py run pixel_llm.programs.hello_world:main
    python pxos_shim.py run --cartridge pxos_v1_1_0.pxa pixel_llm.tests:run_all
    python pxos_shim.py status
    python pxos_shim.py test --cartridge pxos_v1_1_0.pxa
    python pxos_shim.py promote pxos_v1_1_0.pxa

This implements Genesis §2 (Archive-Based Distribution) and §4 (Hypervisor Contract).
"""

import sys
import argparse
from pathlib import Path

# Add pixel_llm to path for hypervisor and cartridge manager
sys.path.insert(0, str(Path(__file__).parent))

from pixel_llm.core.cartridge_manager import get_manager, get_current_cartridge
from pixel_llm.core.hypervisor import get_hypervisor, Hypervisor


def cmd_run(args):
    """Run a program from a cartridge"""
    # Determine which cartridge to use
    if args.cartridge:
        cartridge_name = args.cartridge
        # Verify it exists
        manager = get_manager()
        if not manager.get_cartridge_info(cartridge_name):
            print(f"❌ Cartridge '{cartridge_name}' not found")
            print(f"   Run: python pxos_shim.py status")
            sys.exit(1)
        cartridge_path = Path("pixel_archives") / cartridge_name
        sandbox = True  # Non-current cartridges run in sandbox
    else:
        cartridge_name = get_current_cartridge()
        if not cartridge_name:
            print("❌ No current cartridge set")
            print("   Set one with: python pxos_shim.py promote <name>")
            sys.exit(1)
        cartridge_path = Path("pixel_archives") / cartridge_name if cartridge_name != "development" else None
        sandbox = False

    print(f"🚀 Loading pxOS from: {cartridge_name}")
    if sandbox:
        print("   Running in SANDBOX mode")

    # Initialize hypervisor
    hyper = Hypervisor(cartridge_path=cartridge_path, sandbox=sandbox)

    # Run the program
    program = args.program
    result = hyper.run_program(program)

    if result["success"]:
        print(f"\n✅ Program completed in {result['execution_time']:.3f}s")
        if result.get("result") is not None:
            print(f"   Result: {result['result']}")
        sys.exit(0)
    else:
        print(f"\n❌ Program failed: {result['error']}")
        if args.verbose:
            print("\n" + result.get("traceback", ""))
        sys.exit(1)


def cmd_status(args):
    """Show cartridge status"""
    manager = get_manager()
    manager.print_status()

    if args.verbose:
        print("\n📋 Full Cartridge List:")
        cartridges = manager.list_cartridges()
        for c in cartridges:
            print(f"\n  {c['name']}")
            print(f"    Status: {c['status']}")
            print(f"    Version: {c['version']}")
            print(f"    Generation: {c['generation']}")
            print(f"    Built by: {c['built_by']} ({c['builder_name']})")
            if c.get('parent'):
                print(f"    Parent: {c['parent']}")
            print(f"    Notes: {c['notes']}")


def cmd_test(args):
    """Test Genesis compliance of a cartridge"""
    cartridge_name = args.cartridge

    if not cartridge_name:
        cartridge_name = get_current_cartridge()
        if not cartridge_name:
            print("❌ No cartridge specified and no current cartridge")
            sys.exit(1)

    print(f"🧪 Testing cartridge: {cartridge_name}")

    # Load in sandbox mode
    cartridge_path = Path("pixel_archives") / cartridge_name
    hyper = Hypervisor(cartridge_path=cartridge_path, sandbox=True)

    # Run Genesis validation
    print("\n🔍 Running Genesis compliance checks...")
    result = hyper.validate_genesis()

    print(f"\n📊 Results:")
    print(f"   Compliant: {'✅ YES' if result['compliant'] else '❌ NO'}")
    print(f"   Genesis version: {result['version']}")
    print(f"   Tests passed: {result['tests_passed']}")
    print(f"   Tests failed: {result['tests_failed']}")

    if result['violations']:
        print(f"\n⚠️  Violations:")
        for v in result['violations']:
            print(f"   - {v}")

    # Mark as tested in cartridge manager
    manager = get_manager()
    manager.mark_tested(
        cartridge_name,
        genesis_compliant=result['compliant'],
        test_results=result
    )

    if result['compliant']:
        print(f"\n✅ {cartridge_name} is Genesis compliant")
        print(f"   Ready for promotion with: python pxos_shim.py promote {cartridge_name}")
        sys.exit(0)
    else:
        print(f"\n❌ {cartridge_name} is NOT Genesis compliant")
        print(f"   Fix violations before promotion")
        sys.exit(1)


def cmd_promote(args):
    """Promote a cartridge to current"""
    cartridge_name = args.cartridge

    manager = get_manager()
    info = manager.get_cartridge_info(cartridge_name)

    if not info:
        print(f"❌ Cartridge '{cartridge_name}' not found")
        sys.exit(1)

    # Check if tested
    if not info['compliance']['tested'] and not args.force:
        print(f"⚠️  Warning: {cartridge_name} has not been tested")
        print(f"   Run: python pxos_shim.py test --cartridge {cartridge_name}")
        print(f"   Or use --force to skip testing")
        sys.exit(1)

    # Get reason
    reason = args.reason or "Manual promotion via CLI"

    # Promote
    success = manager.promote_cartridge(
        cartridge_name,
        approved_by=args.approved_by or "cli_user",
        reason=reason,
        force=args.force
    )

    if success:
        print(f"\n✅ Promoted {cartridge_name} to current")
        print(f"   Run with: python pxos_shim.py run <program>")
        sys.exit(0)
    else:
        print(f"\n❌ Promotion failed")
        sys.exit(1)


def cmd_rollback(args):
    """Rollback to a previous cartridge"""
    cartridge_name = args.cartridge
    reason = args.reason or "Manual rollback via CLI"

    manager = get_manager()

    success = manager.rollback_to(
        cartridge_name,
        approved_by=args.approved_by or "cli_user",
        reason=reason
    )

    if success:
        print(f"\n✅ Rolled back to {cartridge_name}")
        sys.exit(0)
    else:
        print(f"\n❌ Rollback failed")
        sys.exit(1)


def cmd_lineage(args):
    """Show lineage of a cartridge"""
    cartridge_name = args.cartridge or get_current_cartridge()

    if not cartridge_name:
        print("❌ No cartridge specified and no current cartridge")
        sys.exit(1)

    manager = get_manager()
    lineage = manager.get_lineage(cartridge_name)

    print(f"\n📜 Lineage of {cartridge_name}:")
    for i, ancestor in enumerate(lineage):
        indent = "  " * i
        info = manager.get_cartridge_info(ancestor)
        is_current = (i == len(lineage) - 1)
        marker = "🎯" if is_current else "└─"
        print(f"{indent}{marker} {ancestor} (gen {info['generation']})")
        print(f"{indent}   v{info['version']} by {info['builder_name']}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="pxOS Launcher - Boot into Pixel World",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pxos_shim.py run pixel_llm.programs.hello_world:main
  python pxos_shim.py run --cartridge pxos_v1_1_0.pxa pixel_llm.tests:run_all
  python pxos_shim.py status
  python pxos_shim.py test --cartridge pxos_v1_1_0.pxa
  python pxos_shim.py promote pxos_v1_1_0.pxa
  python pxos_shim.py rollback pxos_v1_0_0.pxa
  python pxos_shim.py lineage
        """
    )

    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run a program')
    run_parser.add_argument('program', help='Program to run (module:function)')
    run_parser.add_argument('--cartridge', help='Specific cartridge to use (default: current)')
    run_parser.set_defaults(func=cmd_run)

    # Status command
    status_parser = subparsers.add_parser('status', help='Show cartridge status')
    status_parser.set_defaults(func=cmd_status)

    # Test command
    test_parser = subparsers.add_parser('test', help='Test Genesis compliance')
    test_parser.add_argument('--cartridge', help='Cartridge to test (default: current)')
    test_parser.set_defaults(func=cmd_test)

    # Promote command
    promote_parser = subparsers.add_parser('promote', help='Promote cartridge to current')
    promote_parser.add_argument('cartridge', help='Cartridge to promote')
    promote_parser.add_argument('--reason', help='Reason for promotion')
    promote_parser.add_argument('--approved-by', help='Who approved this')
    promote_parser.add_argument('--force', action='store_true', help='Skip testing')
    promote_parser.set_defaults(func=cmd_promote)

    # Rollback command
    rollback_parser = subparsers.add_parser('rollback', help='Rollback to previous version')
    rollback_parser.add_argument('cartridge', help='Cartridge to rollback to')
    rollback_parser.add_argument('--reason', help='Reason for rollback')
    rollback_parser.add_argument('--approved-by', help='Who approved this')
    rollback_parser.set_defaults(func=cmd_rollback)

    # Lineage command
    lineage_parser = subparsers.add_parser('lineage', help='Show cartridge lineage')
    lineage_parser.add_argument('cartridge', nargs='?', help='Cartridge (default: current)')
    lineage_parser.set_defaults(func=cmd_lineage)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run the command
    args.func(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
