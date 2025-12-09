"""CLI commands for ddpc-data."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ddpc.data._cli_base import FriendlyCommand, FriendlyGroup

console = Console()


@click.group(cls=FriendlyGroup, invoke_without_command=True)
@click.version_option(version="2026.1.0", prog_name="ddpc-data")
@click.pass_context
def cli(ctx):
    """Electronic structure data I/O tools."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@cli.group(cls=FriendlyGroup)
def band():
    """Band structure commands."""


@cli.group(cls=FriendlyGroup)
def dos():
    """Density of states commands."""


@band.command(cls=FriendlyCommand)
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", help="Output file path (CSV format)")
@click.option("--mode", default=5, type=int, help="Projection mode (default: 5)")
@click.option("--format", default="csv", type=click.Choice(["csv", "npz"]), help="Output format")
def read(input_file, output, mode, format):
    """Read band structure data and export."""
    from ddpc.data import read_band, to_csv, to_npz

    console.print(f"[cyan]Reading band structure:[/cyan] {input_file}")
    console.print(f"[cyan]Projection mode:[/cyan] {mode}")

    try:
        data, efermi, isproj = read_band(input_file, mode=mode)

        console.print(f"[green]Fermi energy:[/green] {efermi:.4f} eV")
        console.print(f"[green]Has projections:[/green] {isproj}")
        console.print(f"[green]Data columns:[/green] {len(data)}")
        console.print(f"[green]K-points:[/green] {len(data.get('dist', []))}")

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "csv":
                to_csv(data, output_path)
            elif format == "npz":
                to_npz(data, output_path)

            console.print(f"[bold green]✓[/bold green] Saved to: {output_path}")
        else:
            # Show sample data
            table = Table(title="Band Data Preview")
            # Show first few columns
            col_names = list(data.keys())[:5]
            for col in col_names:
                table.add_column(col, style="cyan")

            # Show first 5 rows
            for i in range(min(5, len(data.get("dist", [])))):
                row = [
                    str(data[col][i] if hasattr(data[col], "__getitem__") else data[col])
                    for col in col_names
                ]
                table.add_row(*row)

            console.print(table)
            console.print("[yellow]Use -o/--output to save data[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort from None


@band.command(cls=FriendlyCommand)
@click.argument("input_file", type=click.Path(exists=True))
def info(input_file):
    """Display band structure metadata."""
    from ddpc.data import read_band

    console.print(f"[cyan]Reading band structure info:[/cyan] {input_file}")

    try:
        data, efermi, isproj = read_band(input_file, mode=0)  # mode=0 for total band

        table = Table(title="Band Structure Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("File", str(input_file))
        table.add_row("Fermi Energy", f"{efermi:.6f} eV")
        table.add_row("Has Projections", str(isproj))
        table.add_row("Total Columns", str(len(data)))

        # Count k-points
        if "dist" in data:
            table.add_row("K-points", str(len(data["dist"])))

        # Count bands
        band_cols = [k for k in data.keys() if k.startswith("band")]
        if band_cols:
            table.add_row("Number of Bands", str(len(band_cols)))

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort from None


@dos.command(cls=FriendlyCommand)
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", help="Output file path (CSV format)")
@click.option("--mode", default=5, type=int, help="Projection mode (default: 5)")
@click.option("--format", default="csv", type=click.Choice(["csv", "npz"]), help="Output format")
def read(input_file, output, mode, format):
    """Read density of states data and export."""
    from ddpc.data import read_dos, to_csv, to_npz

    console.print(f"[cyan]Reading DOS:[/cyan] {input_file}")
    console.print(f"[cyan]Projection mode:[/cyan] {mode}")

    try:
        data, efermi, isproj = read_dos(input_file, mode=mode)

        console.print(f"[green]Fermi energy:[/green] {efermi:.4f} eV")
        console.print(f"[green]Has projections:[/green] {isproj}")
        console.print(f"[green]Data columns:[/green] {len(data)}")
        console.print(f"[green]Energy points:[/green] {len(data.get('energy', []))}")

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if format == "csv":
                to_csv(data, output_path)
            elif format == "npz":
                to_npz(data, output_path)

            console.print(f"[bold green]✓[/bold green] Saved to: {output_path}")
        else:
            # Show sample data
            table = Table(title="DOS Data Preview")
            col_names = list(data.keys())[:5]
            for col in col_names:
                table.add_column(col, style="cyan")

            # Show first 5 rows
            for i in range(min(5, len(data.get("energy", [])))):
                row = [
                    f"{data[col][i]:.4f}" if hasattr(data[col], "__getitem__") else str(data[col])
                    for col in col_names
                ]
                table.add_row(*row)

            console.print(table)
            console.print("[yellow]Use -o/--output to save data[/yellow]")

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort from None


@dos.command(cls=FriendlyCommand)
@click.argument("input_file", type=click.Path(exists=True))
def info(input_file):
    """Display DOS metadata."""
    from ddpc.data import read_dos

    console.print(f"[cyan]Reading DOS info:[/cyan] {input_file}")

    try:
        data, efermi, isproj = read_dos(input_file, mode=0)  # mode=0 for total DOS

        table = Table(title="DOS Information")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("File", str(input_file))
        table.add_row("Fermi Energy", f"{efermi:.6f} eV")
        table.add_row("Has Projections", str(isproj))
        table.add_row("Total Columns", str(len(data)))

        if "energy" in data:
            energies = data["energy"]
            table.add_row("Energy Points", str(len(energies)))
            table.add_row("Energy Range", f"{energies.min():.2f} to {energies.max():.2f} eV")

        console.print(table)

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise click.Abort from None


if __name__ == "__main__":
    cli()
