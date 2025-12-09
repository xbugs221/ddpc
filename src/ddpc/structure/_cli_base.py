"""Shared CLI base classes for all DDPC packages."""

import sys

import click
from rich.console import Console

# Force UTF-8 encoding on Windows to handle Unicode characters
console = Console(force_terminal=True, legacy_windows=False)


class FriendlyCommand(click.Command):
    """Custom Click Command that shows help instead of error on incorrect parameters."""

    def main(self, *args, **kwargs):
        """Override main to catch usage errors and show help instead."""
        # Force standalone_mode=False to catch exceptions
        kwargs["standalone_mode"] = False
        try:
            return super().main(*args, **kwargs)
        except click.UsageError as e:
            # Display help instead of error message
            ctx = getattr(e, "ctx", None)
            if ctx:
                console.print(ctx.get_help())
            else:
                console.print(self.get_help(click.Context(self)))
            sys.exit(2)


class FriendlyGroup(click.Group):
    """Custom Click Group that shows help instead of error on incorrect parameters."""

    def main(self, *args, **kwargs):
        """Override main to catch usage errors and show help instead."""
        # Force standalone_mode=False to catch exceptions
        kwargs["standalone_mode"] = False
        try:
            result = super().main(*args, **kwargs)
            return result
        except click.UsageError as e:
            # Display help instead of error message
            ctx = getattr(e, "ctx", None)
            if ctx:
                console.print(ctx.get_help())
            else:
                console.print(self.get_help(click.Context(self)))
            # Return 0 if it's just missing subcommand (not a real error)
            sys.exit(0 if "Missing command" in str(e) else 2)
