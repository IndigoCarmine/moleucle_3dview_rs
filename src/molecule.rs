use crate::atom_radii::vdw_radius;
use crate::spatial_grid::SpatialGrid;
use crate::ANGSTROM_TO_NM as ANGSTROM_TO_NANOMETER;
use lin_alg::f32::Vec3;
use std::path::Path;

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
    /// Build an element from a symbol, trimming surrounding whitespace.
    ///
    /// Trimming here rather than at each call site keeps the 3-byte budget for
    /// the symbol itself — fixed-column formats pad their element field, and
    /// `" C "` would otherwise fill the whole buffer with one useful character.
    pub fn new(symbol: &str) -> Self {
        let src = symbol.trim().as_bytes();
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

/// An interned name: an index into a [`Molecule`]'s [`SymbolTable`].
///
/// Atom and residue names come from a handful of distinct strings repeated
/// across the whole system ("OW", "SOL", "CA", "ALA"). Storing an index rather
/// than a `String` per atom is what keeps [`Atom`] a 32-byte `Copy` value with
/// no heap allocation of its own: a solvated 200k-atom system paid ~110 bytes
/// and three allocations per atom for names alone before this.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct SymbolId(u32);

impl SymbolId {
    /// No name at all — distinct from an interned empty string, which a source
    /// that writes a blank field still produces.
    pub const NONE: SymbolId = SymbolId(u32::MAX);

    pub fn is_none(self) -> bool {
        self == Self::NONE
    }
}

impl Default for SymbolId {
    fn default() -> Self {
        Self::NONE
    }
}

/// The distinct name strings of one molecule, each stored once.
///
/// Built as atoms are pushed through [`MoleculeBuilder`] and then read-only for
/// the molecule's lifetime. `index` exists only to deduplicate during the build;
/// it is kept afterwards so a molecule can still be extended, and costs nothing
/// worth reclaiming — a system of any size has at most a few hundred distinct
/// atom and residue names.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SymbolTable {
    symbols: Vec<Box<str>>,
    index: std::collections::HashMap<Box<str>, SymbolId>,
}

impl SymbolTable {
    /// Intern `text`, returning the id it is stored under. Repeated calls with
    /// the same text return the same id and allocate nothing.
    pub fn intern(&mut self, text: &str) -> SymbolId {
        if let Some(&id) = self.index.get(text) {
            return id;
        }
        let id = SymbolId(self.symbols.len() as u32);
        let boxed: Box<str> = text.into();
        self.symbols.push(boxed.clone());
        self.index.insert(boxed, id);
        id
    }

    /// The text behind `id`, or `None` for [`SymbolId::NONE`] (and for an id
    /// from a different molecule's table, which is out of range here).
    pub fn resolve(&self, id: SymbolId) -> Option<&str> {
        self.symbols.get(id.0 as usize).map(|s| &**s)
    }

    /// How many distinct strings are interned.
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

/// Optional per-atom attributes, as handed to [`MoleculeBuilder::push`].
///
/// Transient and borrowed, never stored: the builder interns the strings into
/// the molecule's [`SymbolTable`] and packs the rest into the atom itself, so a
/// parser can pass slices of the line it is already holding and nothing here
/// outlives the call.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct AtomMeta<'a> {
    pub name: Option<&'a str>,     // Atom identifier (e.g., "CA", "C00")
    pub res_name: Option<&'a str>, // Residue name (e.g., "ALA")
    pub chain_id: Option<char>,    // Chain identifier (e.g., 'A')
    pub res_seq: Option<i32>,      // Residue sequence number
    pub occupancy: Option<f32>,    // Occupancy factor (0.0-1.0)
    pub temp_factor: Option<f32>,  // Temperature factor
    pub charge: Option<&'a str>,   // Formal charge
}

/// One atom: 32 bytes, `Copy`, and with no heap allocation of its own.
///
/// Names live in the owning [`Molecule`]'s [`SymbolTable`], so reading them
/// goes through the molecule ([`Molecule::name_of`], [`Molecule::res_name_of`])
/// rather than the atom. The two attributes that fit inline — the residue
/// sequence number and the chain id — are answered by [`Atom`] directly. The
/// three PDB-only floats/strings (occupancy, temperature factor, formal charge)
/// sit in columns on the molecule that stay empty for sources that have none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Atom {
    pub position: Vec3,
    /// Residue sequence number, or [`Atom::NO_RES_SEQ`] when the source had none.
    res_seq: i32,
    pub element: Element,
    name: SymbolId,
    res_name: SymbolId,
    /// Chain identifier as one ASCII byte; `0` when the source had none.
    chain_id: u8,
}

impl Atom {
    /// Sentinel stored in `res_seq` for "no residue number". `i32::MIN` is not a
    /// value any coordinate format can express in its residue field.
    const NO_RES_SEQ: i32 = i32::MIN;

    /// An atom with only a position and an element — no names, no residue.
    /// The minimal atom a caller can synthesize without a [`MoleculeBuilder`].
    pub fn new(position: Vec3, element: Element) -> Self {
        Atom {
            position,
            res_seq: Self::NO_RES_SEQ,
            element,
            name: SymbolId::NONE,
            res_name: SymbolId::NONE,
            chain_id: 0,
        }
    }

    /// This atom's interned name id. Resolve it against the owning molecule's
    /// [`Molecule::symbols`]; [`Molecule::name_of`] does both in one step.
    pub fn name_id(&self) -> SymbolId {
        self.name
    }

    /// This atom's interned residue-name id. See [`Molecule::res_name_of`].
    pub fn res_name_id(&self) -> SymbolId {
        self.res_name
    }

    pub fn chain_id(&self) -> Option<char> {
        (self.chain_id != 0).then_some(self.chain_id as char)
    }

    pub fn res_seq(&self) -> Option<i32> {
        (self.res_seq != Self::NO_RES_SEQ).then_some(self.res_seq)
    }
}

/// One bond: 12 bytes. The endpoints are `u32` because a bond list is as long
/// as the atom list on a fully-bonded system, and no coordinate format can
/// address more than 4 billion atoms — `usize` endpoints doubled the cost of
/// every bond for range nobody can reach.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Bond {
    pub atom_a: u32,
    pub atom_b: u32,
    pub order: u8,
}

impl Bond {
    /// A single bond between two 0-based atom indices.
    pub fn new(atom_a: usize, atom_b: usize, order: u8) -> Self {
        Bond {
            atom_a: atom_a as u32,
            atom_b: atom_b as u32,
            order,
        }
    }

    /// The endpoints as indices into [`Molecule::atoms`].
    #[inline]
    pub fn endpoints(&self) -> (usize, usize) {
        (self.atom_a as usize, self.atom_b as usize)
    }
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
    /// Every distinct atom/residue/charge string in this molecule, stored once.
    symbols: SymbolTable,
    /// PDB-only per-atom columns. Empty unless the source carried the field, in
    /// which case there is one entry per atom. Keeping them out of [`Atom`] is
    /// what lets a GRO or MOL2 system pay nothing for fields it does not have.
    occupancy: Vec<f32>,
    temp_factor: Vec<f32>,
    charge: Vec<SymbolId>,
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

    /// This molecule's interned strings. Needed to turn an [`Atom`]'s
    /// [`SymbolId`] into text; [`Self::name_of`] and [`Self::res_name_of`] are
    /// the usual way in.
    pub fn symbols(&self) -> &SymbolTable {
        &self.symbols
    }

    /// The atom's name, or `None` when its source carried none.
    pub fn name_of(&self, atom: &Atom) -> Option<&str> {
        self.symbols.resolve(atom.name)
    }

    /// The atom's residue name, or `None` when its source carried none.
    pub fn res_name_of(&self, atom: &Atom) -> Option<&str> {
        self.symbols.resolve(atom.res_name)
    }

    /// The name of the atom at `index`, or `None` when it has none or the index
    /// is out of range.
    pub fn atom_name(&self, index: usize) -> Option<&str> {
        self.name_of(self.atoms.get(index)?)
    }

    /// The residue name of the atom at `index`. See [`Self::atom_name`].
    pub fn atom_res_name(&self, index: usize) -> Option<&str> {
        self.res_name_of(self.atoms.get(index)?)
    }

    /// Occupancy of the atom at `index`, for sources that record one. Matches
    /// the PDB convention that a non-positive occupancy means "not stated".
    pub fn occupancy(&self, index: usize) -> Option<f32> {
        self.occupancy.get(index).copied().filter(|v| *v > 0.0)
    }

    /// Temperature (B) factor of the atom at `index`, where one was recorded.
    pub fn temp_factor(&self, index: usize) -> Option<f32> {
        self.temp_factor.get(index).copied().filter(|v| *v > 0.0)
    }

    /// Formal charge of the atom at `index` as written by its source (e.g.
    /// `"1+"`), where one was recorded.
    pub fn charge(&self, index: usize) -> Option<&str> {
        let id = self.charge.get(index).copied()?;
        self.symbols.resolve(id).filter(|s| !s.is_empty())
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

    /// Replace every atom's residue name, interning the new names.
    ///
    /// For callers that re-derive residue names from a second file (a topology
    /// whose `[ atoms ]` names the residues a bare coordinate file does not).
    /// `names` must have one entry per atom; otherwise the molecule is left
    /// unchanged.
    pub fn set_res_names<S: AsRef<str>>(&mut self, names: &[S]) -> Result<(), String> {
        if names.len() != self.atoms.len() {
            return Err(format!(
                "residue-name count {} does not match atom count {}",
                names.len(),
                self.atoms.len()
            ));
        }
        for (atom, name) in self.atoms.iter_mut().zip(names) {
            atom.res_name = self.symbols.intern(name.as_ref());
        }
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
    /// Atoms built this way carry no names (see [`Atom::new`]); use
    /// [`Molecule::builder`] when the caller has residue or atom names to keep.
    pub fn from_atoms_bonds(atoms: Vec<Atom>, bonds: Vec<Bond>) -> Self {
        Self {
            atoms,
            bonds,
            ..Self::default()
        }
    }

    /// Start building a molecule whose atoms carry names.
    ///
    /// The builder owns the [`SymbolTable`], so every name is interned as it is
    /// pushed and the finished molecule holds one copy of each distinct string.
    pub fn builder() -> MoleculeBuilder {
        MoleculeBuilder::default()
    }

    /// A builder with room for `atoms` atoms reserved up front.
    pub fn builder_with_capacity(atoms: usize) -> MoleculeBuilder {
        MoleculeBuilder::with_capacity(atoms)
    }

    /// Like [`from_atoms_bonds`](Self::from_atoms_bonds) but infers bonds from
    /// interatomic distance (covalent-ish cutoff, in nanometers) when the caller
    /// has no explicit connectivity — handy for display of geometry built by a
    /// pipeline that doesn't track bonds. `cutoff` is the maximum bonded
    /// distance; pairs closer than that (and not the same atom) get a single bond.
    pub fn from_atoms_inferred_bonds(atoms: Vec<Atom>, cutoff: f32) -> Self {
        let cutoff_sq = cutoff * cutoff;
        let mut bonds = Vec::new();

        // A cell of `cutoff` guarantees any pair within `cutoff` lands in the
        // same or an adjacent cell, so the 3x3x3 search is exact.
        let grid = SpatialGrid::points(&atoms, cutoff.max(1e-4));
        for i in 0..atoms.len() {
            let pos_i = atoms[i].position;
            grid.for_each_near(pos_i, |candidate| {
                let j = candidate as usize;
                if j <= i {
                    return;
                }
                let d = pos_i - atoms[j].position;
                let dist_sq = d.magnitude_squared();
                // Guard against coincident atoms producing spurious zero-length bonds.
                if dist_sq > 1e-8 && dist_sq < cutoff_sq {
                    bonds.push(Bond::new(i, j, 1));
                }
            });
        }

        Self::from_atoms_bonds(atoms, bonds)
    }

    /// Resolve a bond's endpoint positions, or `None` if either index is out of
    /// range.
    ///
    /// `atoms` and `bonds` are public fields, and callers routinely assemble a
    /// molecule from a topology file and a coordinate file that disagree on the
    /// atom count — so a bond pointing past the atom list is a normal input, not
    /// a caller bug. Every render and picking loop goes through here so that
    /// case degrades to a missing stick instead of panicking mid-repaint.
    #[inline]
    pub fn bond_endpoints(&self, bond: &Bond) -> Option<(Vec3, Vec3)> {
        let (a, b) = bond.endpoints();
        Some((self.atoms.get(a)?.position, self.atoms.get(b)?.position))
    }

    /// Report the bonds whose endpoints are out of range, as `(index, bond)`.
    ///
    /// Rendering and picking already skip these ([`Self::bond_endpoints`]), so
    /// this is for callers that want to *tell the user* their topology and
    /// coordinates disagree rather than silently dropping the connectivity.
    pub fn invalid_bonds(&self) -> impl Iterator<Item = (usize, &Bond)> {
        let atom_count = self.atoms.len();
        self.bonds
            .iter()
            .enumerate()
            .filter(move |(_, bond)| {
                let (a, b) = bond.endpoints();
                a >= atom_count || b >= atom_count
            })
    }

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

                            atoms.push(Atom::new(
                                Vec3::new(
                                    x * ANGSTROM_TO_NANOMETER,
                                    y * ANGSTROM_TO_NANOMETER,
                                    z * ANGSTROM_TO_NANOMETER,
                                ),
                                Element::new(&element),
                            ));
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
                            bonds.push(Bond::new(a_id - 1, b_id - 1, order));
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(Molecule {
            atoms,
            bonds,
            ..Molecule::default()
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
        let mut builder = Molecule::builder();
        let mut conect_bonds = Vec::with_capacity(256);

        // PDB is a fixed-column format, so every field here is a byte range.
        // Slice with `get`, never `[..]`: a stray non-ASCII byte anywhere in
        // these columns would otherwise split a UTF-8 character and panic on a
        // file the user merely opened.
        for line in content.lines() {
            match line.get(..std::cmp::min(6, line.len())) {
                Some("ATOM  ") | Some("HETATM") => {
                    if let Some(record) = AtomRecord::from_line(line) {
                        push_pdb_record(&mut builder, &record);
                    }
                }
                Some("CONECT") => {
                    // Parse CONECT records for explicit bond information
                    let Some(atom1) = line
                        .get(6..11)
                        .and_then(|field| field.trim().parse::<usize>().ok())
                        .filter(|serial| *serial > 0)
                    else {
                        continue;
                    };

                    // CONECT records can have multiple bonded atoms
                    for i in 0..4 {
                        let start = 11 + i * 5;
                        let Some(atom2) = line
                            .get(start..start + 5)
                            .and_then(|field| field.trim().parse::<usize>().ok())
                        else {
                            continue;
                        };

                        if atom2 > 0 && atom1 < atom2 {
                            // Convert to 0-based
                            conect_bonds.push((atom1 - 1, atom2 - 1));
                        }
                    }
                }
                _ => {}
            }
        }

        // Use explicit bonds if available, otherwise infer from distances
        if conect_bonds.is_empty() {
            return Ok(builder.finish_with_inferred_bonds());
        }

        let atom_count = builder.len();
        let mut bonds = Vec::with_capacity(conect_bonds.len());
        for (a, b) in conect_bonds {
            if a < atom_count && b < atom_count {
                bonds.push(Bond::new(a, b, 1));
            }
        }
        Ok(builder.finish(bonds))
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

            atoms.push(Atom::new(
                Vec3::new(x, y, z),
                Element::new(&element_from_gro_name(name)),
            ));
        }

        Ok(Molecule {
            atoms,
            bonds: Vec::new(),
            ..Molecule::default()
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

        // Precompute radii once so the inner loop is pure arithmetic.
        let radii: Vec<f32> = atoms.iter().map(|a| vdw_radius(&a.element)).collect();
        let max_radius = radii.iter().fold(0.0_f32, |m, &r| m.max(r));

        // Two atoms can only bond if their centers are within
        // (radius_i + radius_j) * factor. The largest possible such distance is
        // 2 * max_radius * factor, so a cell of that size guarantees every
        // bonded pair lands in the same or an adjacent cell.
        let cell_size = (2.0 * max_radius * BOND_DISTANCE_FACTOR).max(MIN_DISTANCE);
        let grid = SpatialGrid::points(atoms, cell_size);

        let mut bonds = Vec::with_capacity(atoms.len() * 2); // ~2 bonds per atom

        for i in 0..atoms.len() {
            let pos_i = atoms[i].position;
            let radius_i = radii[i];

            grid.for_each_near(pos_i, |candidate| {
                let j = candidate as usize;
                // Each unordered pair is emitted once, preserving atom_a < atom_b.
                if j <= i {
                    return;
                }

                let diff = atoms[j].position - pos_i;
                let dist_sq = diff.x * diff.x + diff.y * diff.y + diff.z * diff.z;

                if dist_sq < MIN_DISTANCE_SQ {
                    return;
                }

                let expected_dist = radius_i + radii[j];
                let max_dist_sq = expected_dist * expected_dist * BOND_DISTANCE_FACTOR_SQ;

                if dist_sq < max_dist_sq {
                    bonds.push(Bond::new(i, j, 1));
                }
            });
        }

        bonds
    }
}

/// Assembles a [`Molecule`] atom by atom, interning names as it goes.
///
/// Parsers push one atom at a time with a borrowed [`AtomMeta`]; the builder
/// interns the strings into the molecule's shared [`SymbolTable`] and
/// materializes the PDB-only columns only if some atom actually supplies them.
/// A GRO or MOL2 system therefore never allocates for occupancy, temperature
/// factor or charge at all.
#[derive(Debug, Default)]
pub struct MoleculeBuilder {
    mol: Molecule,
}

impl MoleculeBuilder {
    pub fn with_capacity(atoms: usize) -> Self {
        Self {
            mol: Molecule {
                atoms: Vec::with_capacity(atoms),
                ..Molecule::default()
            },
        }
    }

    /// Append one atom, returning its 0-based index.
    pub fn push(&mut self, position: Vec3, element: &str, meta: &AtomMeta<'_>) -> usize {
        let index = self.mol.atoms.len();
        let name = match meta.name {
            Some(text) => self.mol.symbols.intern(text),
            None => SymbolId::NONE,
        };
        let res_name = match meta.res_name {
            Some(text) => self.mol.symbols.intern(text),
            None => SymbolId::NONE,
        };
        // A chain id is one ASCII character by the PDB spec; anything else (and
        // the blank field a parser reports as `None`) becomes "no chain".
        let chain_id = meta
            .chain_id
            .filter(|c| c.is_ascii() && *c != '\0')
            .map(|c| c as u8)
            .unwrap_or(0);

        self.mol.atoms.push(Atom {
            position,
            res_seq: meta.res_seq.unwrap_or(Atom::NO_RES_SEQ),
            element: Element::new(element),
            name,
            res_name,
            chain_id,
        });

        // The three optional columns stay empty until an atom supplies one, at
        // which point the earlier atoms are backfilled with "not stated".
        if let Some(occupancy) = meta.occupancy {
            self.mol.occupancy.resize(index, 0.0);
            self.mol.occupancy.push(occupancy);
        } else if !self.mol.occupancy.is_empty() {
            self.mol.occupancy.push(0.0);
        }
        if let Some(temp_factor) = meta.temp_factor {
            self.mol.temp_factor.resize(index, 0.0);
            self.mol.temp_factor.push(temp_factor);
        } else if !self.mol.temp_factor.is_empty() {
            self.mol.temp_factor.push(0.0);
        }
        if let Some(charge) = meta.charge {
            let id = self.mol.symbols.intern(charge);
            self.mol.charge.resize(index, SymbolId::NONE);
            self.mol.charge.push(id);
        } else if !self.mol.charge.is_empty() {
            self.mol.charge.push(SymbolId::NONE);
        }

        index
    }

    /// The atoms pushed so far, for callers that need to infer connectivity
    /// from geometry before finishing.
    pub fn atoms(&self) -> &[Atom] {
        &self.mol.atoms
    }

    pub fn len(&self) -> usize {
        self.mol.atoms.len()
    }

    pub fn is_empty(&self) -> bool {
        self.mol.atoms.is_empty()
    }

    /// Intern a string into the molecule being built, for callers that keep
    /// their own name ids alongside the atoms.
    pub fn intern(&mut self, text: &str) -> SymbolId {
        self.mol.symbols.intern(text)
    }

    /// Finish with an explicit bond list.
    pub fn finish(mut self, bonds: Vec<Bond>) -> Molecule {
        self.mol.bonds = bonds;
        self.mol
    }

    /// Finish with bonds inferred from van der Waals radii, the way
    /// [`Molecule::from_pdb`] does when a file has no CONECT records.
    pub fn finish_with_inferred_bonds(mut self) -> Molecule {
        self.mol.bonds = Molecule::infer_bonds(&self.mol.atoms);
        self.mol
    }
}

/// Push one parsed PDB record onto `builder`, applying the PDB conventions for
/// "field not stated" (a blank chain, a non-positive occupancy or B-factor, an
/// empty charge) so those never reach the molecule's optional columns.
fn push_pdb_record(builder: &mut MoleculeBuilder, record: &AtomRecord) {
    let element = extract_element_symbol(&record.element, &record.name);
    builder.push(
        Vec3::new(
            record.x * ANGSTROM_TO_NANOMETER,
            record.y * ANGSTROM_TO_NANOMETER,
            record.z * ANGSTROM_TO_NANOMETER,
        ),
        &element,
        &AtomMeta {
            name: Some(record.name.as_str()),
            res_name: Some(record.res_name.as_str()),
            chain_id: (record.chain_id != ' ').then_some(record.chain_id),
            res_seq: Some(record.res_seq),
            occupancy: (record.occupancy > 0.0).then_some(record.occupancy),
            temp_factor: (record.temp_factor > 0.0).then_some(record.temp_factor),
            charge: (!record.charge.is_empty()).then_some(record.charge.as_str()),
        },
    );
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
        Atom::new(Vec3::new(x, y, z), Element::new(element))
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
                    bonds.push(Bond::new(i, j, 1));
                }
            }
        }
        bonds
    }

    fn sorted_pairs(bonds: &[Bond]) -> Vec<(usize, usize)> {
        let mut pairs: Vec<(usize, usize)> = bonds.iter().map(Bond::endpoints).collect();
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
    fn element_symbols_are_trimmed_on_construction() {
        // Fixed-column formats pad the element field; a padded symbol must not
        // eat the whole 3-byte inline budget.
        assert_eq!(Element::new(" C ").as_str(), "C");
        assert_eq!(Element::new("CL ").as_str(), "CL");
        assert_eq!(Element::new("  ").as_str(), "");
    }

    #[test]
    fn bond_endpoints_reject_out_of_range_indices() {
        let mol = Molecule::from_atoms_bonds(
            vec![atom_at("C", 0.0, 0.0, 0.0), atom_at("O", 0.12, 0.0, 0.0)],
            vec![
                Bond::new(0, 1, 1),
                // Topology and coordinates disagreeing on the atom count is a
                // routine input, not a caller bug -- it must not panic.
                Bond::new(0, 99, 1),
                Bond::new(u32::MAX as usize, 0, 1),
            ],
        );

        assert!(mol.bond_endpoints(&mol.bonds[0]).is_some());
        assert!(mol.bond_endpoints(&mol.bonds[1]).is_none());
        assert!(mol.bond_endpoints(&mol.bonds[2]).is_none());

        let invalid: Vec<usize> = mol.invalid_bonds().map(|(i, _)| i).collect();
        assert_eq!(invalid, vec![1, 2]);
    }

    /// PDB is a fixed-column format, so the parser slices by byte offset. A
    /// non-ASCII byte landing in one of those columns used to split a UTF-8
    /// character and panic on a file the user merely opened.
    #[test]
    fn from_pdb_survives_non_ascii_in_fixed_columns() {
        use std::io::Write;

        let mut file = tempfile::NamedTempFile::new().expect("temp file");
        writeln!(file, "HEADER    a molecule").unwrap();
        // Multi-byte characters straddling the 0..6 record-name columns, the
        // 6..11 CONECT serial columns, and the 11..16 partner columns.
        writeln!(file, "\u{e9}\u{e9}\u{e9}TAM  1  X").unwrap();
        writeln!(file, "CONECT\u{e9}\u{e9}\u{e9}   2").unwrap();
        writeln!(file, "CONECT    1\u{e9}\u{e9}\u{e9}\u{e9}\u{e9}").unwrap();
        writeln!(
            file,
            "ATOM      1  C   MOL A   1       1.000   2.000   3.000  1.00  0.00           C"
        )
        .unwrap();
        writeln!(
            file,
            "ATOM      2  O   MOL A   1       2.000   2.000   3.000  1.00  0.00           O"
        )
        .unwrap();
        writeln!(file, "\u{3042}").unwrap();
        file.flush().unwrap();

        let mol = Molecule::from_pdb(file.path()).expect("non-ASCII lines should be skipped");
        assert_eq!(mol.atoms.len(), 2);
        assert!(mol.invalid_bonds().next().is_none());
    }

    #[test]
    fn set_positions_updates_in_place_and_bumps_generation() {
        let mut mol = Molecule {
            atoms: vec![atom_at("C", 0.0, 0.0, 0.0), atom_at("O", 1.0, 0.0, 0.0)],
            bonds: vec![Bond::new(0, 1, 1)],
            ..Molecule::default()
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
        let mut mol =
            Molecule::from_atoms_bonds(vec![atom_at("C", 0.0, 0.0, 0.0)], Vec::new());
        // Five in-place updates, so the generation counter is non-zero.
        for _ in 0..5 {
            mol.set_positions(&[Vec3::new(0.0, 0.0, 0.0)]).unwrap();
        }
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
        assert!(mol.atom_name(0).is_none());
        assert!(mol.atom_res_name(0).is_none());
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
            ..Molecule::default()
        };
        mol.set_positions_angstrom(&[[10.0, 20.0, 30.0]]).unwrap();
        let p = mol.atoms[0].position;
        assert!((p.x - 1.0).abs() < 1e-6);
        assert!((p.y - 2.0).abs() < 1e-6);
        assert!((p.z - 3.0).abs() < 1e-6);
    }
}
