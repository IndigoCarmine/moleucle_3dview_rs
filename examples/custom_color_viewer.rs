/// Example demonstrating custom color function for atom coloring
/// 
/// This example shows how to use a custom color function to control
/// atom colors dynamically based on element type and selection state.

use moleucle_3dview_rs::{Atom, Molecule, MoleculeViewer, SelectedAtomRender};
use std::path::Path;

/// Custom color function: color atoms by atomic number (alternative color scheme)
fn color_by_atomic_number(atom: &Atom, is_selected: bool) -> (f32, f32, f32) {
    if is_selected {
        return (1.0, 1.0, 0.0); // Yellow for selected atoms
    }

    match atom.element.as_str() {
        "H" => (1.0, 1.0, 1.0),    // White
        "C" => (0.2, 0.2, 0.2),    // Dark gray
        "N" => (0.2, 0.2, 0.8),    // Blue
        "O" => (0.8, 0.2, 0.2),    // Red
        "F" => (0.2, 0.9, 0.2),    // Green
        "P" => (1.0, 0.5, 0.0),    // Orange
        "S" => (1.0, 1.0, 0.2),    // Yellow
        "Cl" => (0.2, 0.9, 0.2),   // Green
        "Br" => (0.6, 0.2, 0.1),   // Brown
        "I" => (0.4, 0.0, 0.4),    // Purple
        _ => (0.5, 0.5, 0.5),      // Gray for unknown
    }
}

/// Custom color function: color atoms by size (CPK coloring style)
fn color_by_van_der_waals(atom: &Atom, is_selected: bool) -> (f32, f32, f32) {
    if is_selected {
        return (1.0, 1.0, 0.0); // Yellow for selected atoms
    }

    match atom.element.as_str() {
        "H" => (1.0, 1.0, 1.0),    // White (smallest)
        "C" => (0.0, 0.0, 0.0),    // Black
        "N" => (0.0, 0.0, 1.0),    // Blue
        "O" => (1.0, 0.0, 0.0),    // Red
        "S" => (1.0, 1.0, 0.0),    // Yellow
        "P" => (1.0, 0.5, 0.0),    // Orange
        _ => (0.5, 0.5, 0.5),      // Gray
    }
}

/// Custom gradient-based color function based on element
fn color_by_electronegativity(atom: &Atom, is_selected: bool) -> (f32, f32, f32) {
    if is_selected {
        return (1.0, 1.0, 0.0); // Yellow for selected atoms
    }

    // Simplified electronegativity coloring: low (red) to high (blue)
    match atom.element.as_str() {
        "H" => (0.8, 0.8, 0.8),    // Neutral
        "C" => (0.4, 0.4, 0.4),    // Low-medium
        "N" => (0.3, 0.3, 0.8),    // High
        "O" => (0.8, 0.2, 0.2),    // Very high
        "F" => (0.2, 0.2, 1.0),    // Highest
        "S" => (1.0, 1.0, 0.2),    // Low-medium
        "P" => (1.0, 0.6, 0.0),    // Low
        "Cl" => (0.2, 0.8, 0.2),   // High
        _ => (0.5, 0.5, 0.5),      // Unknown
    }
}

fn main() {
    println!("Custom Color Function Example");
    println!("=====================================");

    // Example 1: Using default color function
    println!("\n1. Creating viewer with default color function...");
    let mut viewer1: MoleculeViewer<SelectedAtomRender> = MoleculeViewer::new();
    println!("   Viewer created with default coloring (by element)");

    // Example 2: Creating viewer with custom color function at construction
    println!("\n2. Creating viewer with custom color function...");
    let mut viewer2: MoleculeViewer<SelectedAtomRender> =
        MoleculeViewer::with_color_fn(color_by_atomic_number);
    println!("   Viewer created with atomic number-based coloring");

    // Example 3: Changing color function after creation
    println!("\n3. Changing color function after creation...");
    viewer1.set_color_fn(color_by_van_der_waals);
    println!("   Color function changed to van der Waals coloring");

    // Example 4: Using another color function
    println!("\n4. Switching to electronegativity-based coloring...");
    viewer1.set_color_fn(color_by_electronegativity);
    println!("   Color function changed to electronegativity coloring");

    // Load a molecule to demonstrate
    println!("\n5. Loading molecule...");
    let path = Path::new("Benzene.mol2");
    if path.exists() {
        match Molecule::from_mol2(path) {
            Ok(mol) => {
                println!("   Loaded molecule with {} atoms", mol.atoms.len());
                viewer1.set_molecule(mol.clone());
                viewer2.set_molecule(mol);

                println!("\nMolecule details:");
                for atom in &viewer1.molecule.as_ref().unwrap().atoms {
                    let color = (color_by_atomic_number)(atom, false);
                    println!(
                        "   Atom {}: {} at ({:.2}, {:.2}, {:.2}) -> Color: ({:.1}, {:.1}, {:.1})",
                        atom.id, atom.element, atom.position.x, atom.position.y, atom.position.z,
                        color.0, color.1, color.2
                    );
                }
            }
            Err(e) => eprintln!("   Failed to load molecule: {}", e),
        }
    } else {
        eprintln!("   Benzene.mol2 not found at {:?}", std::env::current_dir());
    }

    println!("\n=====================================");
    println!("Color function design: F(Atom, IsSelected) -> Color");
    println!("- You can define custom functions that take &Atom and bool");
    println!("- Return a (f32, f32, f32) RGB color tuple");
    println!("- Set via MoleculeViewer::with_color_fn() or set_color_fn()");
}
