use nalgebra::Point3;
use std::path::Path;
use std::fs;

#[derive(Debug, Clone)]
pub struct Atom {
    pub position: Point3<f32>,
    pub element: String,
    pub id: usize,
}

#[derive(Debug, Clone)]
pub struct Bond {
    pub atom_a: usize,
    pub atom_b: usize,
    pub order: u8,
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
            if line.is_empty() { continue; }
            
            if line.starts_with("@<TRIPOS>") {
                section = line;
                continue;
            }
            
            match section {
                "@<TRIPOS>ATOM" => {
                    // id name x y z type ...
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 6 {
                        if let (Ok(x), Ok(y), Ok(z)) = (parts[2].parse::<f32>(), parts[3].parse::<f32>(), parts[4].parse::<f32>()) {
                           // Type often "C.ar", "H", etc. Take first char or split by dot.
                           // let element = parts[1].chars().next().map(|c| c.to_string()).unwrap_or("?".to_string()); // Unused
                           // Better: use the type field parts[5]
                           let type_str = parts[5];
                           let element = type_str.split('.').next().unwrap_or("?").to_uppercase();

                           atoms.push(Atom {
                               position: Point3::new(x, y, z),
                               element,
                               id: atoms.len() + 1, // 1-based usually in file, but we use index
                           });
                        }
                    }
                },
                "@<TRIPOS>BOND" => {
                     // id atom1 atom2 type ...
                     let parts: Vec<&str> = line.split_whitespace().collect();
                     if parts.len() >= 4 {
                         if let (Ok(a_id), Ok(b_id)) = (parts[1].parse::<usize>(), parts[2].parse::<usize>()) {
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
                },
                _ => {}
            }
        }
        
        Ok(Molecule { atoms, bonds })
    }
}
