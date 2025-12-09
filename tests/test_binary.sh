#!/usr/bin/env bash
# Test script for PyInstaller-compiled binary
# Tests all major CLI commands with real test data

set -u  # Exit on undefined variable (but not on command failure)

# Get binary path from argument
BINARY="${1}"
if [ -z "$BINARY" ]; then
    echo "Usage: $0 <binary_path>"
    exit 1
fi

# Check if binary exists
if [ ! -f "$BINARY" ]; then
    echo "ERROR: Binary not found: $BINARY"
    exit 1
fi

# Make binary executable on Unix-like systems
if [ "$(uname)" != "Windows_NT" ]; then
    chmod +x "$BINARY"
fi

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="$SCRIPT_DIR/data/raw"
STRUCTURE_DIR="$SCRIPT_DIR/structure/raw"
OUTPUT_DIR="$(mktemp -d)"

echo "========================================"
echo "Binary Testing Suite"
echo "========================================"
echo "Binary: $BINARY"
echo "Test data: $DATA_DIR"
echo "Structure data: $STRUCTURE_DIR"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Cleanup function
cleanup() {
    echo "Cleaning up temporary files..."
    rm -rf "$OUTPUT_DIR"
}
trap cleanup EXIT

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to run test
run_test() {
    local test_name="$1"
    shift
    echo "Testing: $test_name"

    # Capture output to temporary file
    local output_file="$(mktemp)"
    if "$@" > "$output_file" 2>&1; then
        echo "  ✓ PASSED"
        ((TESTS_PASSED++))
        rm -f "$output_file"
        return 0
    else
        local exit_code=$?
        echo "  ✗ FAILED (exit code: $exit_code)"
        echo "  Error output:"
        sed 's/^/    /' "$output_file"
        ((TESTS_FAILED++))
        rm -f "$output_file"
        return 1
    fi
}

# Helper function to check file exists (doesn't count as separate test)
check_file() {
    local file="$1"
    if [ -f "$file" ]; then
        echo "  ✓ File created: $(basename "$file")"
        return 0
    else
        echo "  ✗ File not created: $(basename "$file")"
        echo "  WARNING: Output file missing!"
        return 1
    fi
}

echo "========================================"
echo "Basic CLI Tests"
echo "========================================"

run_test "CLI version" "$BINARY" --version
run_test "CLI help" "$BINARY" --help

echo ""
echo "========================================"
echo "Data Band Structure Tests"
echo "========================================"

# Test with spinless band data
run_test "Band info (spinless)" \
    "$BINARY" data band info "$DATA_DIR/spinless_band.h5"

run_test "Band read preview (spinless)" \
    "$BINARY" data band read "$DATA_DIR/spinless_band.h5"

run_test "Band export to CSV (spinless)" \
    "$BINARY" data band read "$DATA_DIR/spinless_band.h5" \
    -o "$OUTPUT_DIR/spinless_band.csv" --format csv
check_file "$OUTPUT_DIR/spinless_band.csv"

run_test "Band export to NPZ (spinless)" \
    "$BINARY" data band read "$DATA_DIR/spinless_band.h5" \
    -o "$OUTPUT_DIR/spinless_band.npz" --format npz
check_file "$OUTPUT_DIR/spinless_band.npz"

# Test with collinear band data
run_test "Band info (collinear)" \
    "$BINARY" data band info "$DATA_DIR/collinear_band.h5"

run_test "Band export to CSV (collinear)" \
    "$BINARY" data band read "$DATA_DIR/collinear_band.h5" \
    -o "$OUTPUT_DIR/collinear_band.csv" --format csv
check_file "$OUTPUT_DIR/collinear_band.csv"

# Test with projected band data
run_test "Band info (projected)" \
    "$BINARY" data band info "$DATA_DIR/spinless_pband.h5"

run_test "Band read with projection mode" \
    "$BINARY" data band read "$DATA_DIR/spinless_pband.h5" --mode 0

echo ""
echo "========================================"
echo "Data DOS Tests"
echo "========================================"

# Test with spinless DOS
run_test "DOS info (spinless)" \
    "$BINARY" data dos info "$DATA_DIR/spinless_dos.h5"

run_test "DOS read preview (spinless)" \
    "$BINARY" data dos read "$DATA_DIR/spinless_dos.h5"

run_test "DOS export to CSV (spinless)" \
    "$BINARY" data dos read "$DATA_DIR/spinless_dos.h5" \
    -o "$OUTPUT_DIR/spinless_dos.csv" --format csv
check_file "$OUTPUT_DIR/spinless_dos.csv"

run_test "DOS export to NPZ (spinless)" \
    "$BINARY" data dos read "$DATA_DIR/spinless_dos.h5" \
    -o "$OUTPUT_DIR/spinless_dos.npz" --format npz
check_file "$OUTPUT_DIR/spinless_dos.npz"

# Test with collinear DOS
run_test "DOS info (collinear)" \
    "$BINARY" data dos info "$DATA_DIR/collinear_dos.h5"

run_test "DOS export to CSV (collinear)" \
    "$BINARY" data dos read "$DATA_DIR/collinear_dos.h5" \
    -o "$OUTPUT_DIR/collinear_dos.csv" --format csv
check_file "$OUTPUT_DIR/collinear_dos.csv"

# Test with projected DOS
run_test "DOS info (projected)" \
    "$BINARY" data dos info "$DATA_DIR/spinless_pdos.h5"

echo ""
echo "========================================"
echo "Structure Tests"
echo "========================================"

# Test structure info
run_test "Structure info (VASP)" \
    "$BINARY" structure info "$STRUCTURE_DIR/POSCAR"

run_test "Structure info with symmetry" \
    "$BINARY" structure info "$STRUCTURE_DIR/POSCAR" --show-symmetry

# Test format conversion
run_test "Convert VASP to CIF" \
    "$BINARY" structure convert "$STRUCTURE_DIR/POSCAR" \
    "$OUTPUT_DIR/structure.cif" --format cif
check_file "$OUTPUT_DIR/structure.cif"

run_test "Convert VASP to XYZ" \
    "$BINARY" structure convert "$STRUCTURE_DIR/POSCAR" \
    "$OUTPUT_DIR/structure.xyz" --format xyz
check_file "$OUTPUT_DIR/structure.xyz"

# Note: Cannot convert XYZ without cell info to VASP
# XYZ files from tests don't have cell information

# Test RESCU XYZ format (with cell information)
run_test "Structure info (RESCU XYZ)" \
    "$BINARY" structure info "$STRUCTURE_DIR/graphene.xyz"

run_test "Convert RESCU XYZ to CIF" \
    "$BINARY" structure convert "$STRUCTURE_DIR/graphene.xyz" \
    "$OUTPUT_DIR/graphene.cif" --format cif
check_file "$OUTPUT_DIR/graphene.cif"

# Test DSPAW .as format
run_test "Structure info (DSPAW .as)" \
    "$BINARY" structure info "$STRUCTURE_DIR/all.as"

run_test "Convert DSPAW to VASP" \
    "$BINARY" structure convert "$STRUCTURE_DIR/all.as" \
    "$OUTPUT_DIR/all.vasp"
check_file "$OUTPUT_DIR/all.vasp"

# Test primitive cell (use a structure we can reduce)
run_test "Find primitive cell" \
    "$BINARY" structure primitive "$STRUCTURE_DIR/POSCAR" \
    -o "$OUTPUT_DIR/primitive.vasp"
check_file "$OUTPUT_DIR/primitive.vasp"

# Test scale to fractional coordinates
run_test "Scale to fractional coordinates" \
    "$BINARY" structure scale "$STRUCTURE_DIR/POSCAR_scaled" \
    -o "$OUTPUT_DIR/scaled.vasp"
check_file "$OUTPUT_DIR/scaled.vasp"

echo ""
echo "========================================"
echo "Error Handling Tests"
echo "========================================"

# Test invalid command (should fail)
if "$BINARY" invalid_command 2>/dev/null; then
    echo "Testing: invalid command"
    echo "  ✗ FAILED: Should have failed on invalid command"
    ((TESTS_FAILED++))
else
    echo "Testing: invalid command"
    echo "  ✓ PASSED: Correctly rejected invalid command"
    ((TESTS_PASSED++))
fi

# Test nonexistent file (should fail)
if "$BINARY" data band info /nonexistent/file.h5 2>/dev/null; then
    echo "Testing: nonexistent file"
    echo "  ✗ FAILED: Should have failed on nonexistent file"
    ((TESTS_FAILED++))
else
    echo "Testing: nonexistent file"
    echo "  ✓ PASSED: Correctly rejected nonexistent file"
    ((TESTS_PASSED++))
fi

echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total tests: $((TESTS_PASSED + TESTS_FAILED))"
echo "Passed: $TESTS_PASSED"
echo "Failed: $TESTS_FAILED"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed!"
    exit 1
fi
