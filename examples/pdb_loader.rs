/// Example demonstrating PDB file loading
/// 
/// This example shows how to load molecules from PDB format files.
/// PDB files typically don't contain explicit bond information,
/// so bonds are inferred based on van der Waals radii.

use moleucle_3dview_rs::Molecule;
use std::path::Path;

fn main() {
    println!("PDB File Loading Example");
    println!("=====================================");

    // Try to load A.pdb
    let pdb_path = Path::new("A.pdb");
    
    if pdb_path.exists() {
        println!("\nLoading PDB file: {:?}", pdb_path);
        
        match Molecule::from_pdb(pdb_path) {
            Ok(mol) => {
                println!("✓ Successfully loaded PDB file");
                println!("  Atoms: {}", mol.atoms.len());
                println!("  Bonds (inferred): {}", mol.bonds.len());
                
                println!("\nAtom details:");
                for atom in mol.atoms.iter().take(5) {
                    println!(
                        "  ID: {}, Element: {}, Position: ({:.3}, {:.3}, {:.3})",
                        atom.id, atom.element, atom.position.x, atom.position.y, atom.position.z
                    );
                }
                if mol.atoms.len() > 5 {
                    println!("  ... and {} more atoms", mol.atoms.len() - 5);
                }
                
                println!("\nBond details (first 5):");
                for bond in mol.bonds.iter().take(5) {
                    println!(
                        "  Bond: Atom {} - Atom {} (order: {})",
                        bond.atom_a, bond.atom_b, bond.order
                    );
                }
                if mol.bonds.len() > 5 {
                    println!("  ... and {} more bonds", mol.bonds.len() - 5);
                }
            }
            Err(e) => {
                eprintln!("✗ Failed to load PDB file: {}", e);
            }
        }
    } else {
        eprintln!("✗ A.pdb not found at current directory");
    }

    // Also try to load Benzene.mol2 for comparison
    println!("\n-------------------------------------");
    let mol2_path = Path::new("Benzene.mol2");
    
    if mol2_path.exists() {
        println!("\nLoading MOL2 file: {:?}", mol2_path);
        
        match Molecule::from_mol2(mol2_path) {
            Ok(mol) => {
                println!("✓ Successfully loaded MOL2 file");
                println!("  Atoms: {}", mol.atoms.len());
                println!("  Bonds: {}", mol.bonds.len());
            }
            Err(e) => {
                eprintln!("✗ Failed to load MOL2 file: {}", e);
            }
        }
    } else {
        println!("Benzene.mol2 not found (optional)");
    }
    
    println!("\n=====================================");
    println!("Supported formats:");
    println!("  - PDB (.pdb): Bonds are automatically inferred from atomic distances");
    println!("  - MOL2 (.mol2): Bonds are read from file");
}
