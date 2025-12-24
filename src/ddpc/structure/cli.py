"""CLI commands for ddpc-structure."""

from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from ddpc.structure._cli_base import FriendlyCommand, FriendlyGroup

# Force UTF-8 encoding on Windows to handle Unicode characters
console = Console(force_terminal=True, legacy_windows=False)


@click.group(cls=FriendlyGroup, invoke_without_command=True)
@click.version_option(version="2026.1.0", prog_name="ddpc-structure")
@click.pass_context
def cli(ctx):
    """Crystal structure I/O and manipulation tools."""
    if ctx.invoked_subcommand is None:
        console.print(ctx.get_help())


@cli.command(cls=FriendlyCommand)
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
@click.option("--format", help="Output format (auto-detected if omitted)")
@click.option("--vasp5", is_flag=True, help="Use VASP 5.x format")
@click.option("--direct/--cartesian", default=True, help="Fractional/Cartesian coordinates (VASP)")
def convert(input_file, output_file, format, vasp5, direct):
    """Convert structure file formats."""
    from ddpc.structure import read_structure, write_structure

    console.print(f"[cyan]Reading:[/cyan] {input_file}")
    atoms = read_structure(input_file)
    console.print(f"[green]Formula:[/green] {atoms.get_chemical_formula()}")
    console.print(f"[green]Atoms:[/green] {len(atoms)}")

    kwargs = {}
    if format == "vasp" or output_file.endswith((".vasp", "POSCAR")):
        if vasp5:
            kwargs["vasp5"] = True
        kwargs["direct"] = direct

    write_structure(output_file, atoms, format=format, **kwargs)
    console.print(f"[bold green]✓[/bold green] Converted to: {output_file}")


@cli.command(cls=FriendlyCommand)
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--symprec", default=1e-5, help="Symmetry precision (default: 1e-5)")
@click.option(
    "--angle-tolerance", default=-1.0, help="Angle tolerance in degrees (default: -1.0, auto)"
)
@click.option("--hall-number", default=0, help="Hall symbol serial number (default: 0)")
@click.option(
    "--symbol-type", default=0, help="Symbol type: 0=international, 1=Schoenflies (default: 0)"
)
@click.option("--show-symmetry", is_flag=True, help="Display symmetry information")
def info(input_file, symprec, angle_tolerance, hall_number, symbol_type, show_symmetry):  # noqa: PLR0913
    """Display structure information."""
    from ddpc.structure import get_symmetry, read_structure

    atoms = read_structure(input_file)

    table = Table(title="Structure Information")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("File", str(input_file))
    table.add_row("Formula", atoms.get_chemical_formula())
    table.add_row("Atoms", str(len(atoms)))

    if atoms.cell.array.any():
        table.add_row("Volume", f"{atoms.get_volume():.2f} Ų")
        lengths = atoms.cell.lengths()
        angles = atoms.cell.angles()
        table.add_row("Cell lengths (Å)", f"{lengths[0]:.2f}, {lengths[1]:.2f}, {lengths[2]:.2f}")
        table.add_row("Cell angles (°)", f"{angles[0]:.1f}, {angles[1]:.1f}, {angles[2]:.1f}")
    else:
        table.add_row("Volume", "Unknown")
        table.add_row("Cell lengths (Å)", "Unknown")
        table.add_row("Cell angles (°)", "Unknown")

    if show_symmetry:
        try:
            sym = get_symmetry(
                atoms,
                symprec=symprec,
                angle_tolerance=angle_tolerance,
                hall_number=hall_number,
                symbol_type=symbol_type,
            )
            table.add_row("Space group", f"{sym['spacegroup']} (#{sym['spacegroup_number']})")
            table.add_row("Crystal system", sym["crystal_system"])
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not determine symmetry: {e}")

    console.print(table)


@cli.command(cls=FriendlyCommand)
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", default="primitive.vasp", help="Output file path")
@click.option("--symprec", default=1e-5, help="Symmetry precision (default: 1e-5)")
@click.option(
    "--angle-tolerance", default=-1.0, help="Angle tolerance in degrees (default: -1.0, auto)"
)
@click.option("--format", help="Output format")
def primitive(input_file, output, symprec, angle_tolerance, format):
    """Find primitive cell."""
    from ddpc.structure import find_primitive, read_structure, write_structure

    console.print(f"[cyan]Finding primitive cell:[/cyan] {input_file}")
    console.print(f"[cyan]Symmetry precision:[/cyan] {symprec}")

    atoms = read_structure(input_file)
    prim = find_primitive(atoms, symprec=symprec, angle_tolerance=angle_tolerance)

    table = Table()
    table.add_column("Structure", style="cyan")
    table.add_column("Atoms", style="green")
    table.add_column("Formula", style="green")
    table.add_column("Volume", style="green")

    table.add_row(
        "Original",
        str(len(atoms)),
        atoms.get_chemical_formula(),
        f"{atoms.get_volume():.2f} Ų",
    )
    table.add_row(
        "Primitive",
        str(len(prim)),
        prim.get_chemical_formula(),
        f"{prim.get_volume():.2f} Ų",
    )

    console.print(table)

    reduction = (1 - len(prim) / len(atoms)) * 100
    console.print(f"[bold green]✓[/bold green] Reduced by {reduction:.1f}%")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_structure(output, prim, format=format)
    console.print(f"[bold green]✓[/bold green] Saved to: {output}")


@cli.command(cls=FriendlyCommand)
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", default="orthogonal.vasp", help="Output file path")
@click.option("--min-atoms", type=int, help="Minimum number of atoms in supercell")
@click.option("--max-atoms", type=int, help="Maximum number of atoms in supercell")
@click.option("--min-length", default=15.0, help="Minimum cell length (default: 15.0 Å)")
@click.option("--max-length", default=20.0, help="Maximum cell length (default: 20.0 Å)")
@click.option("--force-diagonal", is_flag=True, help="Force diagonal lattice vectors")
@click.option("--force-90-degrees", is_flag=True, help="Force 90° angles")
@click.option("--allow-orthorhombic", is_flag=True, help="Allow orthorhombic cells")
@click.option("--angle-tolerance", default=0.001, help="Angle tolerance (default: 0.001)")
@click.option("--step-size", default=0.1, help="Step size for search (default: 0.1)")
@click.option("--format", help="Output format")
def orthogonal(  # noqa: PLR0913
    input_file,
    output,
    min_atoms,
    max_atoms,
    min_length,
    max_length,
    force_diagonal,
    force_90_degrees,
    allow_orthorhombic,
    angle_tolerance,
    step_size,
    format,
):
    """Find orthogonal supercell."""
    from ddpc.structure import find_orthogonal, read_structure, write_structure

    console.print(f"[cyan]Finding orthogonal supercell:[/cyan] {input_file}")
    console.print(f"[cyan]Max cell length:[/cyan] {max_length} Å")

    atoms = read_structure(input_file)

    # Build kwargs dict, only include non-None values
    kwargs = {
        "min_length": min_length,
        "max_length": max_length,
        "angle_tolerance": angle_tolerance,
        "step_size": step_size,
    }
    if min_atoms is not None:
        kwargs["min_atoms"] = min_atoms
    if max_atoms is not None:
        kwargs["max_atoms"] = max_atoms
    if force_diagonal:
        kwargs["force_diagonal"] = force_diagonal
    if force_90_degrees:
        kwargs["force_90_degrees"] = force_90_degrees
    if allow_orthorhombic:
        kwargs["allow_orthorhombic"] = allow_orthorhombic

    ortho = find_orthogonal(atoms, **kwargs)

    table = Table()
    table.add_column("Structure", style="cyan")
    table.add_column("Atoms", style="green")
    table.add_column("Cell angles (°)", style="green")
    table.add_column("Volume (Ų)", style="green")

    orig_angles = atoms.cell.angles()
    ortho_angles = ortho.cell.angles()

    table.add_row(
        "Original",
        str(len(atoms)),
        f"{orig_angles[0]:.1f}, {orig_angles[1]:.1f}, {orig_angles[2]:.1f}",
        f"{atoms.get_volume():.2f}",
    )
    table.add_row(
        "Orthogonal",
        str(len(ortho)),
        f"{ortho_angles[0]:.1f}, {ortho_angles[1]:.1f}, {ortho_angles[2]:.1f}",
        f"{ortho.get_volume():.2f}",
    )

    console.print(table)

    expansion = len(ortho) / len(atoms)
    console.print(f"[bold green]✓[/bold green] Supercell expansion: {expansion:.1f}x")

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_structure(output, ortho, format=format)
    console.print(f"[bold green]✓[/bold green] Saved to: {output}")


@cli.command(cls=FriendlyCommand)
@click.argument("input_file", type=click.Path(exists=True))
@click.option("-o", "--output", default="scaled.vasp", help="Output file path")
def scale(input_file, output):
    """Convert to fractional coordinates."""
    from ddpc.structure import read_structure, scale_positions, write_structure

    console.print(f"[cyan]Converting to fractional coordinates:[/cyan] {input_file}")

    atoms = read_structure(input_file)
    scaled = scale_positions(atoms)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    write_structure(output, scaled, format="vasp", direct=True)
    console.print(f"[bold green]✓[/bold green] Saved to: {output}")


if __name__ == "__main__":
    cli()
