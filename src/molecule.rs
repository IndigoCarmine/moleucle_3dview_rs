use crate::atom_radii::vdw_radius;
use lin_alg::f32::Vec3;
use std::path::Path;

const ANGSTROM_TO_NANOMETER: f32 = 0.1;

/// Compact, `Copy` element symbol stored inline so each atom carries no
/// per-atom heap allocation for its element (the previous `String` cost ~24
/// bytes plus a heap allocation per atom — prohibitive at 500k atoms).
///
/// Chemical symbols are at most a few ASCII characters; longer inputs are
/// truncated, which never happens for real element symbols.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Element {
    bytes: [u8; 3],
    len: u8,
}

impl Element {
    pub fn new(symbol: &str) -> Self {
        let src = symbol.as_bytes();
        let len = src.len().min(3);
        let mut bytes = [0u8; 3];
        bytes[..len].copy_from_slice(&src[..len]);
        Self {
            bytes,
            len: len as u8,
        }
    }

    pub fn as_str(&self) -> &str {
        // Bytes were copied from a valid &str of ASCII element symbols.
        std::str::from_utf8(&self.bytes[..self.len as usize]).unwrap_or("")
    }
}

impl std::ops::Deref for Element {
    type Target = str;
    fn deref(&self) -> &str {
        self.as_str()
    }
}

impl std::fmt::Debug for Element {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(self.as_str(), f)
    }
}

impl std::fmt::Display for Element {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<&str> for Element {
    fn from(s: &str) -> Self {
        Element::new(s)
    }
}

/// Optional PDB-specific attributes. Boxed behind `Atom::meta` so a minimal
/// atom (MOL2, or any source without these fields) stays small and so the
/// per-frame render loops iterate a tight `Atom` array instead of paying for
/// seven mostly-empty `Option`s inline.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AtomMeta {
    pub name: Option<String>,     // Atom identifier (e.g., "CA", "C00")
    pub res_name: Option<String>, // Residue name (e.g., "ALA")
    pub chain_id: Option<char>,   // Chain identifier (e.g., 'A')
    pub res_seq: Option<i32>,     // Residue sequence number
    pub occupancy: Option<f32>,   // Occupancy factor (0.0-1.0)
    pub temp_factor: Option<f32>, // Temperature factor
    pub charge: Option<String>,   // Formal charge
}

#[derive(Debug, Clone)]
pub struct Atom {
    pub position: Vec3,
    pub element: Element,
    pub id: usize,
    /// PDB-specific attributes, present only when a source provides them.
    pub meta: Option<Box<AtomMeta>>,
}

impl Atom {
    pub fn name(&self) -> Option<&str> {
        self.meta.as_ref().and_then(|m| m.name.as_deref())
    }

    pub fn res_name(&self) -> Option<&str> {
        self.meta.as_ref().and_then(|m| m.res_name.as_deref())
    }

    pub fn chain_id(&self) -> Option<char> {
        self.meta.as_ref().and_then(|m| m.chain_id)
    }

    pub fn res_seq(&self) -> Option<i32> {
        self.meta.as_ref().and_then(|m| m.res_seq)
    }

    pub fn occupancy(&self) -> Option<f32> {
        self.meta.as_ref().and_then(|m| m.occupancy)
    }

    pub fn temp_factor(&self) -> Option<f32> {
        self.meta.as_ref().and_then(|m| m.temp_factor)
    }

    pub fn charge(&self) -> Option<&str> {
        self.meta.as_ref().and_then(|m| m.charge.as_deref())
    }

    /// Build an atom from a parsed PDB record, moving its owned strings instead
    /// of cloning them. `id` is the atom's 0-based index.
    fn from_record(id: usize, record: AtomRecord) -> Self {
        let element = Element::new(&extract_element_symbol(&record.element, &record.name));
        let chain_id = (record.chain_id != ' ').then_some(record.chain_id);
        let occupancy = (record.occupancy > 0.0).then_some(record.occupancy);
        let temp_factor = (record.temp_factor > 0.0).then_some(record.temp_factor);
        let charge = (!record.charge.is_empty()).then_some(record.charge);

        Atom {
            position: Vec3::new(
                record.x * ANGSTROM_TO_NANOMETER,
                record.y * ANGSTROM_TO_NANOMETER,
                record.z * ANGSTROM_TO_NANOMETER,
            ),
            element,
            id,
            meta: Some(Box::new(AtomMeta {
                name: Some(record.name),
                res_name: Some(record.res_name),
                chain_id,
                res_seq: Some(record.res_seq),
                occupancy,
                temp_factor,
                charge,
            })),
        }
    }
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
    /// Bumped whenever atom positions change in place (e.g. trajectory
    /// playback). Renderers key their cached GPU geometry on this so they
    /// rebuild when the same `Molecule` is mutated rather than replaced.
    generation: u64,
}

impl Molecule {
    /// Monotonic counter that changes whenever positions are updated in place.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Replace every atom's position in place, leaving elements, bonds, ids and
    /// metadata untouched. Intended for trajectory playback: no allocation, no
    /// bond re-inference, and all existing `Atom` storage is reused, so feeding
    /// successive frames of 500k atoms is allocation-free.
    ///
    /// `positions` must contain exactly `self.atoms.len()` entries (already in
    /// the crate's nanometer units); otherwise the molecule is left unchanged.
    pub fn set_positions(&mut self, positions: &[Vec3]) -> Result<(), String> {
        if positions.len() != self.atoms.len() {
            return Err(format!(
                "position count {} does not match atom count {}",
                positions.len(),
                self.atoms.len()
            ));
        }
        for (atom, &pos) in self.atoms.iter_mut().zip(positions) {
            atom.position = pos;
        }
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }

    /// Like [`set_positions`](Self::set_positions) but takes Ångström
    /// coordinates and applies the crate's Å→nm conversion. Useful for feeding
    /// trajectory frames straight from common formats without an intermediate
    /// `Vec<Vec3>`.
    pub fn set_positions_angstrom(&mut self, coords: &[[f32; 3]]) -> Result<(), String> {
        if coords.len() != self.atoms.len() {
            return Err(format!(
                "position count {} does not match atom count {}",
                coords.len(),
                self.atoms.len()
            ));
        }
        for (atom, c) in self.atoms.iter_mut().zip(coords) {
            atom.position = Vec3::new(
                c[0] * ANGSTROM_TO_NANOMETER,
                c[1] * ANGSTROM_TO_NANOMETER,
                c[2] * ANGSTROM_TO_NANOMETER,
            );
        }
        self.generation = self.generation.wrapping_add(1);
        Ok(())
    }

    /// Build a molecule directly from in-memory atoms and bonds (generation 0).
    ///
    /// The file loaders (`from_mol2`/`from_pdb`/`from_gro`) are the usual entry
    /// points, but callers that synthesize geometry programmatically — e.g. a
    /// node-graph builder assembling atoms from its own model — need a way to
    /// construct a `Molecule` without round-tripping through a file. Positions
    /// are taken as-is (the crate's nanometer convention); `bonds` may be empty.
    pub fn from_atoms_bonds(atoms: Vec<Atom>, bonds: Vec<Bond>) -> Self {
        Self {
            atoms,
            bonds,
            generation: 0,
        }
    }

    /// Like [`from_atoms_bonds`](Self::from_atoms_bonds) but infers bonds from
    /// interatomic distance (covalent-ish cutoff, in nanometers) when the caller
    /// has no explicit connectivity — handy for display of geometry built by a
    /// pipeline that doesn't track bonds. `cutoff` is the maximum bonded
    /// distance; pairs closer than that (and not the same atom) get a single bond.
    pub fn from_atoms_inferred_bonds(atoms: Vec<Atom>, cutoff: f32) -> Self {
        let cutoff_sq = cutoff * cutoff;
        let mut bonds = Vec::new();
        for i in 0..atoms.len() {
            for j in (i + 1)..atoms.len() {
                let d = atoms[i].position - atoms[j].position;
                let dist_sq = d.magnitude_squared();
                // Guard against coincident atoms producing spurious zero-length bonds.
                if dist_sq > 1e-8 && dist_sq < cutoff_sq {
                    bonds.push(Bond {
                        atom_a: i,
                        atom_b: j,
                        order: 1,
                    });
                }
            }
        }
        Self::from_atoms_bonds(atoms, bonds)
    }
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

    pub fn radius(&self) -> f32 {
        let center = self.center();
        // return max distance from center to any atom + its van der Waals radius
        self.atoms
            .iter()
            .map(|atom| {
                let dist = (atom.position - center).magnitude();
                dist + vdw_radius(&atom.element)
            })
            .fold(0.0, f32::max)
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
                                element: Element::new(&element),
                                id: atoms.len() + 1,
                                meta: None,
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

        Ok(Molecule {
            atoms,
            bonds,
            generation: 0,
        })
    }

    /// Parse a PDB file and create a Molecule
    /// Note: If CONECT records exist, they are used for bonds.
    /// Otherwise, bonds are inferred based on atomic distances.
    pub fn from_pdb(path: &Path) -> Result<Self, String> {
        let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        // Parse directly into Atoms so the per-record strings are moved once
        // rather than cloned into a second vector; no intermediate
        // Vec<AtomRecord> is kept alive, halving peak memory for large files.
        let mut atoms: Vec<Atom> = Vec::new();
        let mut conect_bonds = Vec::new();
        conect_bonds.reserve(256);

        for line in content.lines() {
            match &line[..std::cmp::min(6, line.len())] {
                "ATOM  " | "HETATM" => {
                    if let Some(record) = AtomRecord::from_line(line) {
                        atoms.push(Atom::from_record(atoms.len(), record));
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

        Ok(Molecule {
            atoms,
            bonds,
            generation: 0,
        })
    }

    /// Load a molecule, dispatching on the file extension: `.gro` (GROMACS),
    /// `.pdb`, or `.mol2`. Returns an error for unrecognized extensions.
    pub fn load(path: &Path) -> Result<Self, String> {
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .as_deref()
        {
            Some("gro") => Self::from_gro(path),
            Some("pdb") => Self::from_pdb(path),
            Some("mol2") => Self::from_mol2(path),
            other => Err(format!(
                "unsupported molecule file extension: {:?}",
                other.unwrap_or("<none>")
            )),
        }
    }

    /// Parse a GROMACS `.gro` coordinate file.
    ///
    /// GRO files store coordinates already in nanometers (the crate's native
    /// unit) and carry no bond information, so bonds are left empty — at the
    /// multi-million-atom scale typical of GRO systems, inferring bonds would be
    /// prohibitively slow and memory-hungry. To keep memory flat for such large
    /// systems, only positions and element are retained; per-atom residue/name
    /// metadata is intentionally dropped (`Atom::meta` is `None`).
    pub fn from_gro(path: &Path) -> Result<Self, String> {
        let file = std::fs::File::open(path).map_err(|e| e.to_string())?;
        Self::parse_gro(std::io::BufReader::new(file))
    }

    /// Core GRO parser, generic over the reader so it can be unit-tested without
    /// a file. Reads line by line to avoid materializing the whole (potentially
    /// hundreds-of-MB) file in memory at once.
    fn parse_gro<R: std::io::BufRead>(mut reader: R) -> Result<Self, String> {
        let mut line = String::new();

        // Line 1: title (ignored).
        if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            return Err("GRO file is empty".to_string());
        }

        // Line 2: atom count.
        line.clear();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let count: usize = line
            .trim()
            .parse()
            .map_err(|_| format!("invalid GRO atom count: {:?}", line.trim()))?;

        let mut atoms = Vec::with_capacity(count);
        for i in 0..count {
            line.clear();
            if reader.read_line(&mut line).map_err(|e| e.to_string())? == 0 {
                return Err(format!(
                    "GRO ended early: expected {count} atoms, found {i}"
                ));
            }

            // Fixed columns: [0,5) resid, [5,10) resname, [10,15) atom name,
            // [15,20) atom serial. Coordinates follow column 20 and are
            // whitespace-separated (already in nm). Names may run into the
            // serial when fields overflow, so never split the first 20 columns
            // on whitespace.
            let name = line.get(10..15).map(str::trim).unwrap_or("");
            let coords = line.get(20..).ok_or_else(|| {
                format!("GRO atom line {} too short: {:?}", i + 1, line.trim_end())
            })?;
            let mut nums = coords.split_whitespace();
            let x: f32 = parse_gro_coord(nums.next(), i)?;
            let y: f32 = parse_gro_coord(nums.next(), i)?;
            let z: f32 = parse_gro_coord(nums.next(), i)?;

            atoms.push(Atom {
                position: Vec3::new(x, y, z),
                element: Element::new(&element_from_gro_name(name)),
                id: i,
                meta: None,
            });
        }

        Ok(Molecule {
            atoms,
            bonds: Vec::new(),
            generation: 0,
        })
    }

    /// Infer bonds based on van der Waals radii.
    ///
    /// Uses a uniform spatial grid so that each atom is only compared against
    /// atoms in neighboring cells instead of every other atom. This turns the
    /// previous O(n^2) scan into ~O(n) for typical molecular densities, which
    /// is required to handle hundreds of thousands of atoms.
    fn infer_bonds(atoms: &[Atom]) -> Vec<Bond> {
        const BOND_DISTANCE_FACTOR: f32 = 1.6;
        const BOND_DISTANCE_FACTOR_SQ: f32 = BOND_DISTANCE_FACTOR * BOND_DISTANCE_FACTOR;
        const MIN_DISTANCE: f32 = 0.01;
        const MIN_DISTANCE_SQ: f32 = MIN_DISTANCE * MIN_DISTANCE;

        if atoms.len() < 2 {
            return Vec::new();
        }

        // Precompute radii once: vdw_radius() allocates a String per call, so
        // looking it up inside the inner loop would dominate the runtime.
        let radii: Vec<f32> = atoms.iter().map(|a| vdw_radius(&a.element)).collect();
        let max_radius = radii.iter().fold(0.0_f32, |m, &r| m.max(r));

        // Two atoms can only bond if their centers are within
        // (radius_i + radius_j) * factor. The largest possible such distance is
        // 2 * max_radius * factor, so a cell of that size guarantees every
        // bonded pair lands in the same or an adjacent cell.
        let cell_size = (2.0 * max_radius * BOND_DISTANCE_FACTOR).max(MIN_DISTANCE);
        let inv_cell = 1.0 / cell_size;

        // Bounding-box origin so cell indices stay small and non-negative-ish.
        let mut min = atoms[0].position;
        for atom in &atoms[1..] {
            min.x = min.x.min(atom.position.x);
            min.y = min.y.min(atom.position.y);
            min.z = min.z.min(atom.position.z);
        }

        let cell_of = |p: Vec3| -> (i32, i32, i32) {
            (
                ((p.x - min.x) * inv_cell).floor() as i32,
                ((p.y - min.y) * inv_cell).floor() as i32,
                ((p.z - min.z) * inv_cell).floor() as i32,
            )
        };

        // Bucket atom indices by grid cell.
        let mut grid: std::collections::HashMap<(i32, i32, i32), Vec<usize>> =
            std::collections::HashMap::new();
        for (i, atom) in atoms.iter().enumerate() {
            grid.entry(cell_of(atom.position)).or_default().push(i);
        }

        let mut bonds = Vec::with_capacity(atoms.len() * 2); // ~2 bonds per atom
        let mut neighbors: Vec<usize> = Vec::new();

        for i in 0..atoms.len() {
            let pos_i = atoms[i].position;
            let radius_i = radii[i];
            let (cx, cy, cz) = cell_of(pos_i);

            // Gather candidate atoms from the 3x3x3 block of cells around i.
            neighbors.clear();
            for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if let Some(bucket) = grid.get(&(cx + dx, cy + dy, cz + dz)) {
                            neighbors.extend_from_slice(bucket);
                        }
                    }
                }
            }

            for &j in &neighbors {
                // Each unordered pair is emitted once, preserving atom_a < atom_b.
                if j <= i {
                    continue;
                }

                let diff = atoms[j].position - pos_i;
                let dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

                if dist_sq < MIN_DISTANCE_SQ {
                    continue;
                }

                let expected_dist = radius_i + radii[j];
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

fn parse_gro_coord(field: Option<&str>, atom_index: usize) -> Result<f32, String> {
    field
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("GRO atom {} has invalid coordinates", atom_index + 1))
}

/// Derive an element symbol from a GROMACS atom name (e.g. "C1" -> "C",
/// "H14" -> "H", "CL2" -> "CL"). The element is the leading run of letters;
/// only a recognized two-letter element keeps its second letter, so organic
/// names like "C1"/"CA" stay single-letter carbon.
fn element_from_gro_name(name: &str) -> String {
    let letters: String = name
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect::<String>()
        .to_uppercase();

    if letters.is_empty() {
        return "?".to_string();
    }

    // Common two-letter elements that appear in MD systems.
    const TWO_LETTER: &[&str] = &[
        "CL", "BR", "NA", "MG", "CA", "FE", "ZN", "MN", "CU", "NI", "CO", "SI", "SE", "LI", "AL",
        "BA", "SR", "CS", "RB", "KR", "AR", "NE", "HE",
    ];
    if letters.len() >= 2 && TWO_LETTER.contains(&&letters[..2]) {
        letters[..2].to_string()
    } else {
        letters[..1].to_string()
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

#[cfg(test)]
mod tests {
    use super::*;

    fn atom_at(element: &str, x: f32, y: f32, z: f32) -> Atom {
        Atom {
            position: Vec3::new(x, y, z),
            element: Element::new(element),
            id: 0,
            meta: None,
        }
    }

    /// Original O(n^2) reference implementation, kept only for the test.
    fn infer_bonds_bruteforce(atoms: &[Atom]) -> Vec<Bond> {
        const FACTOR_SQ: f32 = 1.6 * 1.6;
        const MIN_DISTANCE_SQ: f32 = 0.01 * 0.01;
        let mut bonds = Vec::new();
        for i in 0..atoms.len() {
            let radius_i = vdw_radius(&atoms[i].element);
            for j in (i + 1)..atoms.len() {
                let diff = atoms[j].position - atoms[i].position;
                let dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;
                if dist_sq < MIN_DISTANCE_SQ {
                    continue;
                }
                let expected = radius_i + vdw_radius(&atoms[j].element);
                if dist_sq < expected * expected * FACTOR_SQ {
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

    fn sorted_pairs(bonds: &[Bond]) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> =
            bonds.iter().map(|b| (b.atom_a, b.atom_b)).collect();
        pairs.sort_unstable();
        pairs
    }

    #[test]
    fn grid_matches_bruteforce_on_small_molecule() {
        // A small grid of carbons spaced ~0.15 nm apart so neighbors bond.
        let mut atoms = Vec::new();
        for ix in 0..4 {
            for iy in 0..4 {
                for iz in 0..4 {
                    atoms.push(atom_at(
                        "C",
                        ix as f32 * 0.15,
                        iy as f32 * 0.15,
                        iz as f32 * 0.15,
                    ));
                }
            }
        }

        let grid = Molecule::infer_bonds(&atoms);
        let brute = infer_bonds_bruteforce(&atoms);
        assert_eq!(sorted_pairs(&grid), sorted_pairs(&brute));
    }

    #[test]
    fn grid_matches_bruteforce_mixed_elements_and_gaps() {
        let atoms = vec![
            atom_at("C", 0.0, 0.0, 0.0),
            atom_at("O", 0.12, 0.0, 0.0),
            atom_at("H", 0.12, 0.10, 0.0),
            // Far away cluster that must not bond to the first.
            atom_at("N", 5.0, 5.0, 5.0),
            atom_at("C", 5.13, 5.0, 5.0),
            // Coincident atoms must be skipped by the MIN_DISTANCE guard.
            atom_at("C", 0.0, 0.0, 0.0),
        ];

        let grid = Molecule::infer_bonds(&atoms);
        let brute = infer_bonds_bruteforce(&atoms);
        assert_eq!(sorted_pairs(&grid), sorted_pairs(&brute));
    }

    #[test]
    fn grid_handles_empty_and_single() {
        assert!(Molecule::infer_bonds(&[]).is_empty());
        assert!(Molecule::infer_bonds(&[atom_at("C", 0.0, 0.0, 0.0)]).is_empty());
    }

    #[test]
    fn set_positions_updates_in_place_and_bumps_generation() {
        let mut mol = Molecule {
            atoms: vec![atom_at("C", 0.0, 0.0, 0.0), atom_at("O", 1.0, 0.0, 0.0)],
            bonds: vec![Bond {
                atom_a: 0,
                atom_b: 1,
                order: 1,
            }],
            generation: 0,
        };

        let new_pos = [Vec3::new(2.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0)];
        mol.set_positions(&new_pos).unwrap();

        assert_eq!(mol.atoms[0].position, new_pos[0]);
        assert_eq!(mol.atoms[1].position, new_pos[1]);
        assert_eq!(mol.generation(), 1);
        // Topology untouched.
        assert_eq!(mol.bonds.len(), 1);
        assert_eq!(mol.atoms[0].element.as_str(), "C");

        mol.set_positions(&new_pos).unwrap();
        assert_eq!(mol.generation(), 2);
    }

    #[test]
    fn set_positions_rejects_length_mismatch() {
        let mut mol = Molecule {
            atoms: vec![atom_at("C", 0.0, 0.0, 0.0)],
            bonds: Vec::new(),
            generation: 5,
        };
        assert!(mol.set_positions(&[]).is_err());
        // Unchanged on error.
        assert_eq!(mol.generation(), 5);
        assert_eq!(mol.atoms[0].position, Vec3::new(0.0, 0.0, 0.0));
    }

    fn gro_line(resid: i32, resname: &str, name: &str, serial: i32, p: [f32; 3]) -> String {
        format!(
            "{:>5}{:<5}{:>5}{:>5}{:8.3}{:8.3}{:8.3}",
            resid, resname, name, serial, p[0], p[1], p[2]
        )
    }

    #[test]
    fn parse_gro_reads_positions_and_elements() {
        let content = format!(
            "title line\n2\n{}\n{}\n   5.00000   5.00000   5.00000\n",
            gro_line(1, "MOL", "C1", 1, [1.0, 2.0, 3.0]),
            gro_line(2, "SOL", "OW", 2, [4.0, 5.0, 6.0]),
        );

        let mol = Molecule::parse_gro(std::io::Cursor::new(content)).unwrap();
        assert_eq!(mol.atoms.len(), 2);
        // GRO coordinates are already in nm — stored verbatim.
        assert_eq!(mol.atoms[0].position, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(mol.atoms[1].position, Vec3::new(4.0, 5.0, 6.0));
        assert_eq!(mol.atoms[0].element.as_str(), "C");
        assert_eq!(mol.atoms[1].element.as_str(), "O");
        // No bonds inferred, no per-atom metadata retained.
        assert!(mol.bonds.is_empty());
        assert!(mol.atoms[0].meta.is_none());
    }

    #[test]
    fn parse_gro_handles_two_letter_element() {
        let content = format!(
            "t\n1\n{}\n   5.0   5.0   5.0\n",
            gro_line(1, "ION", "CL", 1, [0.0, 0.0, 0.0]),
        );
        let mol = Molecule::parse_gro(std::io::Cursor::new(content)).unwrap();
        assert_eq!(mol.atoms[0].element.as_str(), "CL");
    }

    #[test]
    fn parse_gro_rejects_truncated_file() {
        // Header promises 3 atoms but only one is present.
        let content = format!("t\n3\n{}\n", gro_line(1, "MOL", "C1", 1, [0.0, 0.0, 0.0]));
        assert!(Molecule::parse_gro(std::io::Cursor::new(content)).is_err());
    }

    #[test]
    fn set_positions_angstrom_applies_nm_conversion() {
        let mut mol = Molecule {
            atoms: vec![atom_at("C", 0.0, 0.0, 0.0)],
            bonds: Vec::new(),
            generation: 0,
        };
        mol.set_positions_angstrom(&[[10.0, 20.0, 30.0]]).unwrap();
        let p = mol.atoms[0].position;
        assert!((p.x - 1.0).abs() < 1e-6);
        assert!((p.y - 2.0).abs() < 1e-6);
        assert!((p.z - 3.0).abs() < 1e-6);
    }
}
