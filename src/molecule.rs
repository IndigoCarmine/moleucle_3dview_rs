use lin_alg::f32::Vec3;
use std::path::Path;

#[derive(Debug, Clone)]
pub struct Atom {
    pub position: Vec3,
    pub element: String,
    pub id: usize,
    // PDB-specific attributes (optional for MOL2)
    pub name: Option<String>,           // Atom identifier (e.g., "CA", "C00")
    pub res_name: Option<String>,       // Residue name (e.g., "ALA")
    pub chain_id: Option<char>,         // Chain identifier (e.g., 'A')
    pub res_seq: Option<i32>,           // Residue sequence number
    pub occupancy: Option<f32>,         // Occupancy factor (0.0-1.0)
    pub temp_factor: Option<f32>,       // Temperature factor
    pub charge: Option<String>,         // Formal charge
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
    pub fn from_mol2(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        let mut atoms = Vec::new();
        let mut bonds = Vec::new();

        let mut section = "";

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            if line.starts_with("@<TRIPOS>") {
                section = line;
                continue;
            }

            match section {
                "@<TRIPOS>ATOM" => {
                    // id name x y z type ...
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 6 {
                        if let (Ok(x), Ok(y), Ok(z)) = (
                            parts[2].parse::<f32>(),
                            parts[3].parse::<f32>(),
                            parts[4].parse::<f32>(),
                        ) {
                            // Type often "C.ar", "H", etc. Take first char or split by dot.
                            // let element = parts[1].chars().next().map(|c| c.to_string()).unwrap_or("?".to_string()); // Unused
                            // Better: use the type field parts[5]
                            let type_str = parts[5];
                            let element = type_str.split('.').next().unwrap_or("?").to_uppercase();

                            atoms.push(Atom {
                                position: Vec3::new(x, y, z),
                                element,
                                id: atoms.len() + 1, // 1-based usually in file, but we use index
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
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 4 {
                        if let (Ok(a_id), Ok(b_id)) =
                            (parts[1].parse::<usize>(), parts[2].parse::<usize>())
                        {
                            let order = match parts[3] {
                                "2" => 2,
                                "3" => 3,
                                "ar" => 1, // aromatic, often drawn as 1.5 or 1
                                _ => 1,
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

        for line in content.lines() {
            if line.starts_with("ATOM") || line.starts_with("HETATM") {
                if let Some(record) = AtomRecord::from_line(line) {
                    atom_records.push(record);
                }
            } else if line.starts_with("CONECT") {
                // Parse CONECT records for explicit bond information
                if line.len() >= 11 {
                    if let Ok(atom1) = line[6..11].trim().parse::<usize>() {
                        // CONECT records can have multiple bonded atoms
                        for i in 0..4 {
                            let start = 11 + i * 5;
                            let end = start + 5;
                            if end <= line.len() {
                                if let Ok(atom2) = line[start..end].trim().parse::<usize>() {
                                    if atom1 != 0 && atom2 != 0 && atom1 < atom2 {
                                        conect_bonds.push((atom1 - 1, atom2 - 1)); // Convert to 0-based
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Convert AtomRecords to Atoms
        let atoms: Vec<Atom> = atom_records
            .iter()
            .enumerate()
            .map(|(idx, record)| Atom {
                position: Vec3::new(record.x, record.y, record.z),
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
            conect_bonds
                .into_iter()
                .filter(|(a, b)| *a < atoms.len() && *b < atoms.len())
                .map(|(a, b)| Bond {
                    atom_a: a,
                    atom_b: b,
                    order: 1,
                })
                .collect()
        } else {
            Self::infer_bonds(&atoms)
        };

        Ok(Molecule { atoms, bonds })
    }

    /// Infer bonds based on van der Waals radii
    fn infer_bonds(atoms: &[Atom]) -> Vec<Bond> {
        let mut bonds = Vec::new();
        const BOND_DISTANCE_FACTOR: f32 = 1.6; // Tolerance factor for bond detection

        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let dist = (atoms[i].position - atoms[j].position).magnitude();
                let expected_dist = get_vdw_radius(&atoms[i].element)
                    + get_vdw_radius(&atoms[j].element);

                // Check if atoms are within bonding distance
                if dist < expected_dist * BOND_DISTANCE_FACTOR && dist > 0.1 {
                    bonds.push(Bond {
                        atom_a: i,
                        atom_b: j,
                        order: 1, // Default to single bond
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

/// Get approximate van der Waals radius for element
fn get_vdw_radius(element: &str) -> f32 {
    match element.to_uppercase().as_str() {
        "H" => 1.20,
        "C" => 1.70,
        "N" => 1.55,
        "O" => 1.52,
        "F" => 1.47,
        "P" => 1.80,
        "S" => 1.80,
        "Cl" => 1.75,
        "Br" => 1.85,
        "I" => 1.98,
        _ => 1.70, // Default
    }
}
