"""Unified CLI for DDPC packages."""

import sys

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option(version="2026.1.0", prog_name="ddpc")
def cli():
    """DDPC - Crystal structure and electronic data tools.

    This is a unified interface that delegates commands to specialized subpackages:
    - ddpc data: Electronic structure data (band, DOS)
    - ddpc structure: Crystal structure I/O and manipulation
    """


@cli.group()
def data():
    """Electronic structure data commands (requires: ddpc[data])."""


@cli.group()
def structure():
    """Crystal structure commands (requires: ddpc[structure])."""


@data.group()
def band():
    """Band structure commands."""


@data.group()
def dos():
    """Density of states commands."""


def _safe_invoke_command(cli_group, command_name, ctx, package_name, group_name=None):
    """Safely invoke a command from a CLI group with validation."""
    if not hasattr(cli_group, "commands") or command_name not in cli_group.commands:
        console.print(
            f"[bold red]Error:[/bold red] Command [cyan]{command_name}[/cyan] is not available"
        )
        console.print()

        # Show available commands
        if hasattr(cli_group, "commands"):
            available = list(cli_group.commands.keys())
            if available:
                console.print("[dim]Available commands:[/dim]")
                for cmd in sorted(available):
                    console.print(f"  • [cyan]{cmd}[/cyan]")
                console.print()

        # Show help tip
        full_path = f"{package_name} {group_name}" if group_name else package_name
        console.print(f"[dim]For more information, run:[/dim] [cyan]{full_path} --help[/cyan]")
        sys.exit(2)

    command_func = cli_group.commands[command_name]

    # Build correct program name for help display
    # e.g., "ddpc-structure convert" or "ddpc-data band read"
    if group_name:
        prog_name = f"{package_name} {group_name} {command_name}"
    else:
        prog_name = f"{package_name} {command_name}"

    # Call command with correct program name
    # This allows FriendlyCommand to display proper help on errors
    try:
        command_func.main(
            args=ctx.args,
            prog_name=prog_name,
            standalone_mode=True,
        )
    except SystemExit:
        # Re-raise SystemExit to preserve exit codes
        raise


@band.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def read(ctx):
    """Read band structure data (delegates to ddpc-data)."""
    try:
        from ddpc.data.cli import band as band_cli

        _safe_invoke_command(band_cli, "read", ctx, "ddpc-data", "band")
    except ImportError:
        console.print("[bold red]Error:[/bold red] ddpc-data is not installed")
        console.print(
            "Install with: [cyan]pip install ddpc[data][/cyan] "
            "or [cyan]pip install ddpc-data[/cyan]"
        )
        sys.exit(1)


@band.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def info(ctx):
    """Show band structure info (delegates to ddpc-data)."""
    try:
        from ddpc.data.cli import band as band_cli

        _safe_invoke_command(band_cli, "info", ctx, "ddpc-data", "band")
    except ImportError:
        console.print("[bold red]Error:[/bold red] ddpc-data is not installed")
        console.print(
            "Install with: [cyan]pip install ddpc[data][/cyan] "
            "or [cyan]pip install ddpc-data[/cyan]"
        )
        sys.exit(1)


@dos.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def read(ctx):
    """Read DOS data (delegates to ddpc-data)."""
    try:
        from ddpc.data.cli import dos as dos_cli

        _safe_invoke_command(dos_cli, "read", ctx, "ddpc-data", "dos")
    except ImportError:
        console.print("[bold red]Error:[/bold red] ddpc-data is not installed")
        console.print(
            "Install with: [cyan]pip install ddpc[data][/cyan] "
            "or [cyan]pip install ddpc-data[/cyan]"
        )
        sys.exit(1)


@dos.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def info(ctx):
    """Show DOS info (delegates to ddpc-data)."""
    try:
        from ddpc.data.cli import dos as dos_cli

        _safe_invoke_command(dos_cli, "info", ctx, "ddpc-data", "dos")
    except ImportError:
        console.print("[bold red]Error:[/bold red] ddpc-data is not installed")
        console.print(
            "Install with: [cyan]pip install ddpc[data][/cyan] "
            "or [cyan]pip install ddpc-data[/cyan]"
        )
        sys.exit(1)


@structure.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def convert(ctx):
    """Convert structure file formats (delegates to ddpc-structure)."""
    try:
        from ddpc.structure.cli import cli as structure_cli

        _safe_invoke_command(structure_cli, "convert", ctx, "ddpc-structure")
    except ImportError:
        console.print("[bold red]Error:[/bold red] ddpc-structure is not installed")
        console.print(
            "Install with: [cyan]pip install ddpc[structure][/cyan] "
            "or [cyan]pip install ddpc-structure[/cyan]"
        )
        sys.exit(1)


@structure.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def info(ctx):
    """Show structure information (delegates to ddpc-structure)."""
    try:
        from ddpc.structure.cli import cli as structure_cli

        _safe_invoke_command(structure_cli, "info", ctx, "ddpc-structure")
    except ImportError:
        console.print("[bold red]Error:[/bold red] ddpc-structure is not installed")
        console.print(
            "Install with: [cyan]pip install ddpc[structure][/cyan] "
            "or [cyan]pip install ddpc-structure[/cyan]"
        )
        sys.exit(1)


@structure.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def primitive(ctx):
    """Find primitive cell (delegates to ddpc-structure)."""
    try:
        from ddpc.structure.cli import cli as structure_cli

        _safe_invoke_command(structure_cli, "primitive", ctx, "ddpc-structure")
    except ImportError:
        console.print("[bold red]Error:[/bold red] ddpc-structure is not installed")
        console.print(
            "Install with: [cyan]pip install ddpc[structure][/cyan] "
            "or [cyan]pip install ddpc-structure[/cyan]"
        )
        sys.exit(1)


@structure.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def orthogonal(ctx):
    """Find orthogonal supercell (delegates to ddpc-structure)."""
    try:
        from ddpc.structure.cli import cli as structure_cli

        _safe_invoke_command(structure_cli, "orthogonal", ctx, "ddpc-structure")
    except ImportError:
        console.print("[bold red]Error:[/bold red] ddpc-structure is not installed")
        console.print(
            "Install with: [cyan]pip install ddpc[structure][/cyan] "
            "or [cyan]pip install ddpc-structure[/cyan]"
        )
        sys.exit(1)


@structure.command(context_settings={"ignore_unknown_options": True, "allow_extra_args": True})
@click.pass_context
def scale(ctx):
    """Convert to fractional coordinates (delegates to ddpc-structure)."""
    try:
        from ddpc.structure.cli import cli as structure_cli

        _safe_invoke_command(structure_cli, "scale", ctx, "ddpc-structure")
    except ImportError:
        console.print("[bold red]Error:[/bold red] ddpc-structure is not installed")
        console.print(
            "Install with: [cyan]pip install ddpc[structure][/cyan] "
            "or [cyan]pip install ddpc-structure[/cyan]"
        )
        sys.exit(1)


if __name__ == "__main__":
    cli()
