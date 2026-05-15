use crate::atom_radii::vdw_radius;
use lin_alg::f32::Vec3;
use std::path::Path;

const ANGSTROM_TO_NANOMETER: f32 = 0.1;
#[derive(Debug, Clone)]
pub struct Atom {
    pub position: Vec3,
    pub element: String,
    pub id: usize,
    // PDB-specific attributes (optional for MOL2)
    pub name: Option<String>,     // Atom identifier (e.g., "CA", "C00")
    pub res_name: Option<String>, // Residue name (e.g., "ALA")
    pub chain_id: Option<char>,   // Chain identifier (e.g., 'A')
    pub res_seq: Option<i32>,     // Residue sequence number
    pub occupancy: Option<f32>,   // Occupancy factor (0.0-1.0)
    pub temp_factor: Option<f32>, // Temperature factor
    pub charge: Option<String>,   // Formal charge
}

#[derive(Debug, Clone)]
pub struct Bond {
    pub atom_a: usize,
    pub atom_b: usize,
    pub order: u8,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AtomRecord {
    pub serial: usize,
    pub name: String,
    pub alt_loc: char,
    pub res_name: String,
    pub chain_id: char,
    pub res_seq: i32,
    pub i_code: char,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub occupancy: f32,
    pub temp_factor: f32,
    pub element: String,
    pub charge: String,
}

impl AtomRecord {
    pub fn from_line(line: &str) -> Option<Self> {
        if line.len() < 54 {
            return None;
        }

        let parse_int =
            |range: std::ops::Range<usize>| -> Option<i32> { line.get(range)?.trim().parse().ok() };
        let parse_float =
            |range: std::ops::Range<usize>| -> Option<f32> { line.get(range)?.trim().parse().ok() };
        let parse_str = |range: std::ops::Range<usize>| -> String {
            line.get(range)
                .map(|s| s.trim().to_string())
                .unwrap_or_default()
        };
        let parse_char = |index: usize| -> char {
            line.get(index..index + 1)
                .and_then(|s| s.chars().next())
                .unwrap_or(' ')
        };

        Some(AtomRecord {
            serial: parse_int(6..11)? as usize,
            name: parse_str(12..16),
            alt_loc: parse_char(16),
            res_name: parse_str(17..20),
            chain_id: parse_char(21),
            res_seq: parse_int(22..26)?,
            i_code: parse_char(26),
            x: parse_float(30..38)?,
            y: parse_float(38..46)?,
            z: parse_float(46..54)?,
            occupancy: parse_float(54..60).unwrap_or(0.0),
            temp_factor: parse_float(60..66).unwrap_or(0.0),
            element: parse_str(76..78),
            charge: parse_str(78..80),
        })
    }

    pub fn to_line(&self) -> String {
        format!(
            "ATOM  {:5} {:<4}{}{:>3} {}{:4}{:<4}{:8.3}{:8.3}{:8.3}{:6.2}{:6.2}          {:>2}{:>2}",
            self.serial,
            self.name,
            self.alt_loc,
            self.res_name,
            self.chain_id,
            self.res_seq,
            self.i_code,
            self.x,
            self.y,
            self.z,
            self.occupancy,
            self.temp_factor,
            self.element,
            self.charge
        )
    }
}

#[derive(Debug, Clone, Default)]
pub struct Molecule {
    pub atoms: Vec<Atom>,
    pub bonds: Vec<Bond>,
}

impl Molecule {
    pub fn center(&self) -> Vec3 {
        if self.atoms.is_empty() {
            return Vec3::new_zero();
        }

        let sum = self
            .atoms
            .iter()
            .fold(Vec3::new_zero(), |acc, atom| acc + atom.position);
        sum / self.atoms.len() as f32
    }

    pub fn from_mol2(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut atoms = Vec::new();
        let mut bonds = Vec::new();

        // Pre-allocate with reasonable capacity
        atoms.reserve(256); // Common for small-medium molecules
        bonds.reserve(256);

        let mut section = "";

        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed.starts_with("@<TRIPOS>") {
                section = trimmed;
                continue;
            }

            match section {
                "@<TRIPOS>ATOM" => {
                    // id name x y z type ...
                    let mut parts = trimmed.split_whitespace();
                    // Skip id (parts[0])
                    let _ = parts.next();
                    // Skip name (parts[1])
                    let _ = parts.next();

                    if let (Some(x_str), Some(y_str), Some(z_str)) =
                        (parts.next(), parts.next(), parts.next())
                    {
                        if let (Ok(x), Ok(y), Ok(z)) = (
                            x_str.parse::<f32>(),
                            y_str.parse::<f32>(),
                            z_str.parse::<f32>(),
                        ) {
                            let element = parts
                                .next()
                                .and_then(|type_str| type_str.split('.').next())
                                .map(|s| s.to_uppercase())
                                .unwrap_or_else(|| "?".to_string());

                            atoms.push(Atom {
                                position: Vec3::new(
                                    x * ANGSTROM_TO_NANOMETER,
                                    y * ANGSTROM_TO_NANOMETER,
                                    z * ANGSTROM_TO_NANOMETER,
                                ),
                                element,
                                id: atoms.len() + 1,
                                name: None,
                                res_name: None,
                                chain_id: None,
                                res_seq: None,
                                occupancy: None,
                                temp_factor: None,
                                charge: None,
                            });
                        }
                    }
                }
                "@<TRIPOS>BOND" => {
                    // id atom1 atom2 type ...
                    let mut parts = trimmed.split_whitespace();
                    // Skip id (parts[0])
                    let _ = parts.next();

                    let a_id: Option<usize> = parts.next().and_then(|s| s.parse().ok());
                    let b_id: Option<usize> = parts.next().and_then(|s| s.parse().ok());

                    if let (Some(a_id), Some(b_id)) = (a_id, b_id) {
                        let order = match parts.next() {
                            Some("2") => 2u8,
                            Some("3") => 3u8,
                            Some("ar") => 1u8,
                            _ => 1u8,
                        };

                        // Adjust 1-based to 0-based
                        if a_id > 0 && b_id > 0 && a_id <= atoms.len() && b_id <= atoms.len() {
                            bonds.push(Bond {
                                atom_a: a_id - 1,
                                atom_b: b_id - 1,
                                order,
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(Molecule { atoms, bonds })
    }

    /// Parse a PDB file and create a Molecule
    /// Note: If CONECT records exist, they are used for bonds.
    /// Otherwise, bonds are inferred based on atomic distances.
    pub fn from_pdb(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut atom_records = Vec::new();
        let mut conect_bonds = Vec::new();

        // Pre-allocate with typical capacities
        atom_records.reserve(256);
        conect_bonds.reserve(256);

        for line in content.lines() {
            match &line[..std::cmp::min(6, line.len())] {
                "ATOM  " | "HETATM" => {
                    if let Some(record) = AtomRecord::from_line(line) {
                        atom_records.push(record);
                    }
                }
                "CONECT" => {
                    // Parse CONECT records for explicit bond information
                    if line.len() >= 11 {
                        if let Ok(atom1) = line[6..11].trim().parse::<usize>() {
                            if atom1 > 0 {
                                // CONECT records can have multiple bonded atoms
                                for i in 0..4 {
                                    let start = 11 + i * 5;
                                    if start + 5 <= line.len() {
                                        if let Ok(atom2) =
                                            line[start..start + 5].trim().parse::<usize>()
                                        {
                                            if atom2 > 0 && atom1 < atom2 {
                                                conect_bonds.push((atom1 - 1, atom2 - 1));
                                                // Convert to 0-based
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        // Convert AtomRecords to Atoms
        let atoms: Vec<Atom> = atom_records
            .iter()
            .enumerate()
            .map(|(idx, record)| Atom {
                position: Vec3::new(
                    record.x * ANGSTROM_TO_NANOMETER,
                    record.y * ANGSTROM_TO_NANOMETER,
                    record.z * ANGSTROM_TO_NANOMETER,
                ),
                element: extract_element_symbol(&record.element, &record.name),
                id: idx,
                name: Some(record.name.clone()),
                res_name: Some(record.res_name.clone()),
                chain_id: if record.chain_id != ' ' {
                    Some(record.chain_id)
                } else {
                    None
                },
                res_seq: Some(record.res_seq),
                occupancy: if record.occupancy > 0.0 {
                    Some(record.occupancy)
                } else {
                    None
                },
                temp_factor: if record.temp_factor > 0.0 {
                    Some(record.temp_factor)
                } else {
                    None
                },
                charge: if !record.charge.is_empty() {
                    Some(record.charge.clone())
                } else {
                    None
                },
            })
            .collect();

        // Use explicit bonds if available, otherwise infer from distances
        let bonds = if !conect_bonds.is_empty() {
            let mut result = Vec::new();
            result.reserve(conect_bonds.len());
            for (a, b) in conect_bonds {
                if a < atoms.len() && b < atoms.len() {
                    result.push(Bond {
                        atom_a: a,
                        atom_b: b,
                        order: 1,
                    });
                }
            }
            result
        } else {
            Self::infer_bonds(&atoms)
        };

        Ok(Molecule { atoms, bonds })
    }

    /// Infer bonds based on van der Waals radii (optimized)
    fn infer_bonds(atoms: &[Atom]) -> Vec<Bond> {
        let mut bonds = Vec::new();
        const BOND_DISTANCE_FACTOR: f32 = 1.6;
        const BOND_DISTANCE_FACTOR_SQ: f32 = BOND_DISTANCE_FACTOR * BOND_DISTANCE_FACTOR;
        const MIN_DISTANCE: f32 = 0.01;
        const MIN_DISTANCE_SQ: f32 = MIN_DISTANCE * MIN_DISTANCE;

        // Pre-allocate with estimated capacity
        bonds.reserve(atoms.len() * 2); // Typical atom has ~2 bonds

        for i in 0..atoms.len() {
            let pos_i = atoms[i].position;
            let radius_i = vdw_radius(&atoms[i].element);

            for j in (i + 1)..atoms.len() {
                let pos_j = atoms[j].position;
                let diff = pos_j - pos_i;
                let dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

                // Early exit if too far
                if dist_sq < MIN_DISTANCE_SQ {
                    continue;
                }

                let expected_dist = radius_i + vdw_radius(&atoms[j].element);
                let max_dist_sq = expected_dist * expected_dist * BOND_DISTANCE_FACTOR_SQ;

                if dist_sq < max_dist_sq {
                    bonds.push(Bond {
                        atom_a: i,
                        atom_b: j,
                        order: 1,
                    });
                }
            }
        }

        bonds
    }
}

/// Extract element symbol from PDB element string and atom name
fn extract_element_symbol(element: &str, atom_name: &str) -> String {
    // If element field is explicitly provided and not empty, use it
    if !element.is_empty() {
        element.to_uppercase()
    } else {
        // Fallback: extract from atom name (e.g., "CA" -> "C", "HG" -> "H")
        atom_name
            .chars()
            .next()
            .map(|c| c.to_uppercase().collect::<String>())
            .unwrap_or_else(|| "?".to_string())
    }
}
