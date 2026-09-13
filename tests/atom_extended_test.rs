/// Test to verify extended Atom structure with PDB attributes
///
/// Names live in the molecule's shared symbol table, so they are read through
/// the molecule (`name_of`/`res_name_of`) rather than off the atom.
use moleucle_3dview_rs::Molecule;
use std::path::Path;

#[test]
fn test_atom_extended_fields_mol2() {
    let mol = Molecule::from_mol2(Path::new("Benzene.mol2"))
        .expect("Failed to load Benzene.mol2");

    // MOL2 atoms should have None for PDB-specific fields
    for (index, atom) in mol.atoms.iter().enumerate() {
        assert!(mol.name_of(atom).is_none(), "MOL2 atoms should not have name");
        assert!(mol.res_name_of(atom).is_none(), "MOL2 atoms should not have res_name");
        assert!(atom.chain_id().is_none(), "MOL2 atoms should not have chain_id");
        assert!(atom.res_seq().is_none(), "MOL2 atoms should not have res_seq");
        assert!(mol.occupancy(index).is_none(), "MOL2 atoms should not have occupancy");
        assert!(mol.temp_factor(index).is_none(), "MOL2 atoms should not have temp_factor");
        assert!(mol.charge(index).is_none(), "MOL2 atoms should not have charge");
    }
}

#[test]
fn test_atom_extended_fields_pdb() {
    let mol = Molecule::from_pdb(Path::new("A.pdb"))
        .expect("Failed to load A.pdb");

    // PDB atoms should have Some values for extended fields
    for atom in &mol.atoms {
        assert!(
            mol.name_of(atom).is_some(),
            "PDB atoms should have name (e.g., C00, H0C)"
        );
        assert!(
            mol.res_name_of(atom).is_some(),
            "PDB atoms should have res_name (e.g., ENAP)"
        );
        // Note: chain_id might be None if it's a space character
        assert!(
            atom.res_seq().is_some(),
            "PDB atoms should have res_seq (e.g., 1)"
        );
        // occupancy and temp_factor might be None if 0.0
        // charge might be None if empty
    }
}

#[test]
fn test_atom_pdb_name_field() {
    let mol = Molecule::from_pdb(Path::new("A.pdb"))
        .expect("Failed to load A.pdb");

    // First atom should be "C00"
    assert_eq!(
        mol.atom_name(0),
        Some("C00"),
        "First atom in A.pdb should be C00"
    );
}

#[test]
fn test_atom_pdb_residue_info() {
    let mol = Molecule::from_pdb(Path::new("A.pdb"))
        .expect("Failed to load A.pdb");

    // Check residue information consistency
    for atom in &mol.atoms {
        if let Some(res_name) = mol.res_name_of(atom) {
            // A.pdb contains ENAP or ENA residues (PDB parsing may trim differently)
            assert!(
                res_name == "ENAP" || res_name == "ENA",
                "Unexpected residue name: {}",
                res_name
            );
        }

        if let Some(res_seq) = atom.res_seq() {
            assert_eq!(res_seq, 1, "All atoms should have residue sequence 1");
        }
    }
}

#[test]
fn test_atom_extended_structure() {
    let mol = Molecule::from_pdb(Path::new("A.pdb"))
        .expect("Failed to load A.pdb");

    let atom = &mol.atoms[0];

    // Verify all extended fields exist
    assert!(!atom.element.is_empty(), "Element should not be empty");
    assert!(atom.position.x.is_finite());
    
    // Extended fields
    assert!(mol.name_of(atom).is_some());
    assert!(mol.res_name_of(atom).is_some());
    assert!(atom.res_seq().is_some());

    println!(
        "Extended Atom: element={}, name={:?}, res={:?}, seq={:?}",
        atom.element,
        mol.name_of(atom),
        mol.res_name_of(atom),
        atom.res_seq()
    );
}
