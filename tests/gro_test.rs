use moleucle_3dview_rs::Molecule;
use std::path::Path;

/// Parses the bundled sample GROMACS frame if it is present. Ignored by default
/// because the file is large and not committed; run with:
/// `cargo test --test gro_test -- --ignored --nocapture`
#[test]
#[ignore]
fn parse_sample_output_gro() {
    let path = Path::new("output.gro");
    if !path.exists() {
        eprintln!("output.gro not present; skipping");
        return;
    }

    let start = std::time::Instant::now();
    let mol = Molecule::from_gro(path).expect("failed to parse output.gro");
    let elapsed = start.elapsed();

    println!(
        "parsed {} atoms in {:.2}s ({} bonds)",
        mol.atoms.len(),
        elapsed.as_secs_f32(),
        mol.bonds.len(),
    );

    assert_eq!(mol.atoms.len(), 5_420_100, "atom count from header");
    assert!(mol.bonds.is_empty(), "GRO carries no bonds");

    let first = &mol.atoms[0];
    assert_eq!(first.element.as_str(), "C");
    // First atom line: "    1MOL     C1    1  41.690   3.087  27.849" (nm).
    assert!((first.position.x - 41.690).abs() < 1e-3);
    assert!((first.position.y - 3.087).abs() < 1e-3);
    assert!((first.position.z - 27.849).abs() < 1e-3);

    // Every position must be finite (no column-misalignment garbage).
    assert!(mol
        .atoms
        .iter()
        .all(|a| a.position.x.is_finite() && a.position.y.is_finite() && a.position.z.is_finite()));
}
