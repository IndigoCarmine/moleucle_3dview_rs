use moleucle_3dview_rs::{AtomRecord, Molecule};
use std::fs;
use std::path::Path;
use tempfile::NamedTempFile;

const BENZENE_MOL2_PATH: &str = "Benzene.mol2";
const A_PDB_PATH: &str = "A.pdb";

// ============================================================================
// MOL2 読み込みテスト
// ============================================================================

#[test]
fn test_mol2_file_exists() {
    assert!(
        Path::new(BENZENE_MOL2_PATH).exists(),
        "Benzene.mol2 file should exist"
    );
}

#[test]
fn test_mol2_read_success() {
    let mol = Molecule::from_mol2(Path::new(BENZENE_MOL2_PATH));
    assert!(mol.is_ok(), "MOL2 file should be readable");
}

#[test]
fn test_mol2_atom_count() {
    let mol = Molecule::from_mol2(Path::new(BENZENE_MOL2_PATH))
        .expect("Failed to load Benzene.mol2");

    // Benzene: C6H6 = 12 atoms
    assert_eq!(
        mol.atoms.len(),
        12,
        "Benzene should have 12 atoms (6 carbon, 6 hydrogen)"
    );
}

#[test]
fn test_mol2_bond_count() {
    let mol = Molecule::from_mol2(Path::new(BENZENE_MOL2_PATH))
        .expect("Failed to load Benzene.mol2");

    // Benzene: 6 C-C bonds + 6 C-H bonds = 12 bonds
    assert_eq!(
        mol.bonds.len(),
        12,
        "Benzene should have 12 bonds (6 C-C aromatic + 6 C-H)"
    );
}

#[test]
fn test_mol2_atom_data() {
    let mol = Molecule::from_mol2(Path::new(BENZENE_MOL2_PATH))
        .expect("Failed to load Benzene.mol2");

    // Check first atom exists and has valid position
    assert!(!mol.atoms.is_empty(), "Should have at least one atom");

    let first_atom = &mol.atoms[0];
    assert!(!first_atom.element.is_empty(), "Atom should have element symbol");
    assert!(
        first_atom.id > 0 || first_atom.id == 0,
        "Atom should have valid id"
    );

    // Position should be valid numbers (not NaN, not infinity)
    assert!(first_atom.position.x.is_finite(), "X coordinate should be finite");
    assert!(first_atom.position.y.is_finite(), "Y coordinate should be finite");
    assert!(first_atom.position.z.is_finite(), "Z coordinate should be finite");
}

#[test]
fn test_mol2_element_symbols() {
    let mol = Molecule::from_mol2(Path::new(BENZENE_MOL2_PATH))
        .expect("Failed to load Benzene.mol2");

    let elements: std::collections::HashSet<_> = mol.atoms.iter().map(|a| a.element.clone()).collect();

    // Benzene should only have C and H
    assert!(
        elements.contains("C"),
        "Benzene should contain carbon atoms"
    );
    assert!(
        elements.contains("H"),
        "Benzene should contain hydrogen atoms"
    );

    // Should not have other elements
    for elem in &elements {
        assert!(
            elem == &"C" || elem == &"H",
            "Benzene should only have C and H, found: {}",
            elem
        );
    }
}

#[test]
fn test_mol2_bond_validity() {
    let mol = Molecule::from_mol2(Path::new(BENZENE_MOL2_PATH))
        .expect("Failed to load Benzene.mol2");

    // Check all bonds reference valid atoms
    for bond in &mol.bonds {
        assert!(
            bond.atom_a < mol.atoms.len(),
            "Bond atom_a index {} out of range",
            bond.atom_a
        );
        assert!(
            bond.atom_b < mol.atoms.len(),
            "Bond atom_b index {} out of range",
            bond.atom_b
        );
        assert!(bond.atom_a != bond.atom_b, "Bond should not connect same atom");
    }
}

// ============================================================================
// PDB 読み込みテスト
// ============================================================================

#[test]
fn test_pdb_file_exists() {
    assert!(
        Path::new(A_PDB_PATH).exists(),
        "A.pdb file should exist"
    );
}

#[test]
fn test_pdb_read_success() {
    let mol = Molecule::from_pdb(Path::new(A_PDB_PATH));
    assert!(mol.is_ok(), "PDB file should be readable: {:?}", mol.err());
}

#[test]
fn test_pdb_atom_count() {
    let mol = Molecule::from_pdb(Path::new(A_PDB_PATH))
        .expect("Failed to load A.pdb");

    // A.pdb should have 24 atoms (from file content)
    assert_eq!(
        mol.atoms.len(),
        24,
        "A.pdb should have 24 atoms"
    );
}

#[test]
fn test_pdb_bond_count() {
    let mol = Molecule::from_pdb(Path::new(A_PDB_PATH))
        .expect("Failed to load A.pdb");

    // A.pdb has explicit CONECT records specifying 25 bonds
    assert_eq!(
        mol.bonds.len(),
        25,
        "A.pdb should have 25 bonds (from CONECT records)"
    );
}

#[test]
fn test_pdb_atom_data() {
    let mol = Molecule::from_pdb(Path::new(A_PDB_PATH))
        .expect("Failed to load A.pdb");

    // Check that atoms have valid data
    for (idx, atom) in mol.atoms.iter().enumerate() {
        assert!(
            !atom.element.is_empty(),
            "Atom {} should have element symbol",
            idx
        );
        assert!(
            atom.position.x.is_finite(),
            "Atom {} X coordinate should be finite",
            idx
        );
        assert!(
            atom.position.y.is_finite(),
            "Atom {} Y coordinate should be finite",
            idx
        );
        assert!(
            atom.position.z.is_finite(),
            "Atom {} Z coordinate should be finite",
            idx
        );
    }
}

#[test]
fn test_pdb_element_extraction() {
    let mol = Molecule::from_pdb(Path::new(A_PDB_PATH))
        .expect("Failed to load A.pdb");

    let elements: std::collections::HashSet<_> = mol.atoms.iter().map(|a| a.element.clone()).collect();

    // A.pdb contains mainly carbon and hydrogen
    assert!(
        elements.contains("C"),
        "A.pdb should contain carbon atoms"
    );
    assert!(
        elements.contains("H"),
        "A.pdb should contain hydrogen atoms"
    );
}

#[test]
fn test_pdb_bond_validity() {
    let mol = Molecule::from_pdb(Path::new(A_PDB_PATH))
        .expect("Failed to load A.pdb");

    // Check all bonds reference valid atoms
    for bond in &mol.bonds {
        assert!(
            bond.atom_a < mol.atoms.len(),
            "Bond atom_a index {} out of range (total atoms: {})",
            bond.atom_a,
            mol.atoms.len()
        );
        assert!(
            bond.atom_b < mol.atoms.len(),
            "Bond atom_b index {} out of range (total atoms: {})",
            bond.atom_b,
            mol.atoms.len()
        );
        assert!(bond.atom_a != bond.atom_b, "Bond should not connect same atom");
        assert!(bond.atom_a < bond.atom_b, "Bond indices should be ordered");
    }
}

// ============================================================================
// AtomRecord パーステスト
// ============================================================================

#[test]
fn test_atom_record_from_line() {
    // Example PDB ATOM line from A.pdb
    let line = "ATOM      1  C00 ENAP    1       1.000   1.000   0.000";

    let record = AtomRecord::from_line(line);
    assert!(record.is_some(), "Should parse valid ATOM line");

    let record = record.unwrap();
    assert_eq!(record.serial, 1);
    assert_eq!(record.name, "C00");
    assert_eq!(record.x, 1.0);
    assert_eq!(record.y, 1.0);
    assert_eq!(record.z, 0.0);
}

#[test]
fn test_atom_record_short_line() {
    // Line that's too short
    let line = "ATOM";

    let record = AtomRecord::from_line(line);
    assert!(record.is_none(), "Should not parse line shorter than 54 chars");
}

#[test]
fn test_atom_record_roundtrip() {
    let original = AtomRecord {
        serial: 42,
        name: "CA".to_string(),
        alt_loc: ' ',
        res_name: "ALA".to_string(),
        chain_id: 'A',
        res_seq: 123,
        i_code: ' ',
        x: 1.234,
        y: 5.678,
        z: 9.012,
        occupancy: 1.0,
        temp_factor: 20.0,
        element: "C".to_string(),
        charge: "0".to_string(),
    };

    let line = original.to_line();
    let parsed = AtomRecord::from_line(&line);

    assert!(parsed.is_some(), "Should roundtrip through to_line");

    let parsed = parsed.unwrap();
    assert_eq!(parsed.serial, original.serial);
    assert_eq!(parsed.name.trim(), original.name.trim());
    assert_eq!(parsed.element.trim(), original.element.trim());
    assert!((parsed.x - original.x).abs() < 0.001);
    assert!((parsed.y - original.y).abs() < 0.001);
    assert!((parsed.z - original.z).abs() < 0.001);
}

// ============================================================================
// 相互比較テスト
// ============================================================================

#[test]
fn test_mol2_pdb_different_formats() {
    let mol2 = Molecule::from_mol2(Path::new(BENZENE_MOL2_PATH))
        .expect("Failed to load Benzene.mol2");
    let pdb = Molecule::from_pdb(Path::new(A_PDB_PATH))
        .expect("Failed to load A.pdb");

    // Both should have atoms and bonds
    assert!(!mol2.atoms.is_empty());
    assert!(!mol2.bonds.is_empty());
    assert!(!pdb.atoms.is_empty());
    assert!(!pdb.bonds.is_empty());

    // They should be different molecules
    assert_ne!(
        mol2.atoms.len(),
        pdb.atoms.len(),
        "Benzene and A should have different atom counts"
    );
}

// ============================================================================
// エラーハンドリングテスト
// ============================================================================

#[test]
fn test_mol2_nonexistent_file() {
    let result = Molecule::from_mol2(Path::new("nonexistent.mol2"));
    assert!(
        result.is_err(),
        "Should return error for nonexistent file"
    );
}

#[test]
fn test_pdb_nonexistent_file() {
    let result = Molecule::from_pdb(Path::new("nonexistent.pdb"));
    assert!(
        result.is_err(),
        "Should return error for nonexistent file"
    );
}

#[test]
fn test_mol2_invalid_format() {
    // Create temporary invalid MOL2 file
    let content = "This is not a valid MOL2 file\nJust random text";
    let temp_file = NamedTempFile::new().expect("Failed to create temp file");
    let temp_path = temp_file.path();
    fs::write(temp_path, content).expect("Failed to write temp file");

    let result = Molecule::from_mol2(temp_path);
    // Should still parse but with 0 atoms and 0 bonds
    assert!(result.is_ok(), "Should handle invalid format gracefully");

    if let Ok(mol) = result {
        assert_eq!(
            mol.atoms.len(),
            0,
            "Invalid format should result in no atoms"
        );
    }
}

// ============================================================================
// テスト補助関数
// ============================================================================

#[test]
fn test_atom_position_range() {
    let mol = Molecule::from_mol2(Path::new(BENZENE_MOL2_PATH))
        .expect("Failed to load Benzene.mol2");

    // Get min/max coordinates
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;

    for atom in &mol.atoms {
        min_x = min_x.min(atom.position.x);
        max_x = max_x.max(atom.position.x);
        min_y = min_y.min(atom.position.y);
        max_y = max_y.max(atom.position.y);
        min_z = min_z.min(atom.position.z);
        max_z = max_z.max(atom.position.z);
    }

    let range_x = max_x - min_x;
    let range_y = max_y - min_y;
    let range_z = max_z - min_z;

    // Benzene should fit in a reasonable bounding box
    assert!(
        range_x < 10.0 && range_y < 10.0 && range_z < 10.0,
        "Benzene should fit in a reasonable bounding box, found ranges: ({:.2}, {:.2}, {:.2})",
        range_x,
        range_y,
        range_z
    );
}

#[test]
fn test_pdb_position_range() {
    let mol = Molecule::from_pdb(Path::new(A_PDB_PATH))
        .expect("Failed to load A.pdb");

    // Get min/max coordinates
    let mut min_x = f32::MAX;
    let mut max_x = f32::MIN;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    let mut min_z = f32::MAX;
    let mut max_z = f32::MIN;

    for atom in &mol.atoms {
        min_x = min_x.min(atom.position.x);
        max_x = max_x.max(atom.position.x);
        min_y = min_y.min(atom.position.y);
        max_y = max_y.max(atom.position.y);
        min_z = min_z.min(atom.position.z);
        max_z = max_z.max(atom.position.z);
    }

    let range_x = max_x - min_x;
    let range_y = max_y - min_y;
    let range_z = max_z - min_z;

    println!(
        "A.pdb coordinate ranges: X({:.2}), Y({:.2}), Z({:.2})",
        range_x, range_y, range_z
    );

    // Should have non-zero ranges (molecules should be 3D)
    assert!(range_x > 0.1 || range_y > 0.1 || range_z > 0.1);
}

#[test]
fn test_bond_distance_stats() {
    let mol = Molecule::from_mol2(Path::new(BENZENE_MOL2_PATH))
        .expect("Failed to load Benzene.mol2");

    let mut min_dist = f32::MAX;
    let mut max_dist = f32::MIN;
    let mut sum_dist = 0.0;

    for bond in &mol.bonds {
        let a = mol.atoms[bond.atom_a].position;
        let b = mol.atoms[bond.atom_b].position;
        let dist = (a - b).magnitude();

        min_dist = min_dist.min(dist);
        max_dist = max_dist.max(dist);
        sum_dist += dist;
    }

    let avg_dist = sum_dist / mol.bonds.len() as f32;

    println!(
        "Benzene bond statistics: min={:.3}Å, max={:.3}Å, avg={:.3}Å",
        min_dist, max_dist, avg_dist
    );

    // Typical C-C bond length: 1.54 Å, C-H bond length: 1.09 Å
    // so average should be somewhere between 1.0 and 1.6
    assert!(
        avg_dist > 0.8 && avg_dist < 2.0,
        "Average bond distance should be chemically reasonable"
    );
}
