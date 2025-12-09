"""Entry point for ddpc CLI when run as a module or compiled with Nuitka."""

import platform
import shlex
import sys

from ddpc.cli import cli


def run_repl():
    """Run an interactive REPL for the ddpc CLI on Windows."""
    print("Welcome to the ddpc interactive shell.")
    print("You can run any ddpc command here (e.g., 'structure info --help').")
    print("Type 'exit' or 'quit' to leave.")
    print("-" * 30)

    while True:
        try:
            command = input("ddpc> ")
            command = command.strip()
            if not command:
                continue
            if command.lower() in ["quit", "exit"]:
                break

            args = shlex.split(command)
            cli(args)

        except SystemExit as e:
            # Click calls sys.exit() by default. We catch it to keep the REPL running.
            # A non-zero code from sys.exit indicates an error.
            # Click already prints error messages to stderr, so we just continue the loop.
            if e.code is not None and e.code != 0:
                pass  # Click handled the error message.
        except EOFError:
            # This happens on Ctrl+Z -> Enter on Windows, or Ctrl+D on Unix.
            print()  # Print a newline for cleaner exit.
            break
        except Exception:
            # For any other unexpected errors, print them and continue.
            import traceback

            print("An unexpected error occurred:", file=sys.stderr)
            traceback.print_exc()


if __name__ == "__main__":
    is_windows = platform.system() == "Windows"
    # 'frozen' is an attribute set by PyInstaller/Nuitka
    is_frozen = getattr(sys, "frozen", False)
    has_no_args = len(sys.argv) == 1

    # Run REPL only when double-clicked on Windows
    if is_windows and is_frozen and has_no_args:
        run_repl()
    else:
        cli()
