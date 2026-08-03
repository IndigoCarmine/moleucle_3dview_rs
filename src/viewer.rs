use crate::atom_radii::{ball_stick_radius, default_ball_stick_bond_radius};
use crate::molecule::{Atom, Molecule};
use crate::periodic::PeriodicImages;
use crate::spatial_grid::SpatialGrid;
use crate::AdditionalRender;
// Viewer no longer owns shared state; state is passed in by the caller (viewport or user).
use lin_alg::f32::Vec3;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone)]
pub enum ViewerEvent {
    AtomClicked(usize),
    BondClicked(usize),
    NothingClicked,
}

/// Color function type: takes Atom and is_selected flag, returns RGBA color.
/// The fourth component is the alpha channel (1.0 = opaque, 0.0 = transparent).
pub type ColorFn = fn(&Atom, bool) -> (f32, f32, f32, f32);

/// Default color function based on element type. Fully opaque (alpha = 1.0).
///
/// Matching is case-insensitive, like [`crate::vdw_radius`]. The loaders in this
/// crate uppercase element symbols, but callers building `Atom`s by hand are
/// free to use the conventional mixed case ("Cl"), and both must colour the
/// same — an exact-case `"Cl"` arm here was simply unreachable.
pub fn default_color_fn(atom: &Atom, _is_selected: bool) -> (f32, f32, f32, f32) {
    match atom.element.trim().as_bytes() {
        [b'C' | b'c'] => (0.1, 0.1, 0.1, 1.0),                // Black/Dark Grey
        [b'H' | b'h'] => (0.9, 0.9, 0.9, 1.0),                // White
        [b'O' | b'o'] => (0.9, 0.1, 0.1, 1.0),                // Red
        [b'N' | b'n'] => (0.1, 0.1, 0.9, 1.0),                // Blue
        [b'S' | b's'] => (0.9, 0.9, 0.1, 1.0),                // Yellow
        [b'P' | b'p'] => (1.0, 0.6, 0.0, 1.0),                // Orange
        [b'C' | b'c', b'L' | b'l'] => (0.1, 0.9, 0.1, 1.0),   // Green
        [b'B' | b'b', b'R' | b'r'] => (0.6, 0.15, 0.1, 1.0),  // Dark red
        [b'I' | b'i'] => (0.55, 0.15, 0.65, 1.0),             // Purple
        [b'F' | b'f'] => (0.55, 0.85, 0.45, 1.0),             // Pale green
        _ => (0.7, 0.7, 0.7, 1.0),                            // Grey
    }
}

pub struct MoleculeViewer {
    pub molecule: Option<Molecule>,
    /// Bumped by every mutator that changes what the built-in molecule
    /// rendering would produce. See [`MoleculeViewer::revision`].
    revision: u64,
    pub additional_render: Vec<Box<dyn AdditionalRender>>,
    pub color_fn: ColorFn,
    /// Opacity of the whole molecule (atoms + bonds) in `0.0..=1.0`, folded into
    /// each geometry color's alpha at render time. `1.0` is fully opaque.
    pub molecule_opacity: f32,
    /// Optional per-atom sphere radius override (atom order). When `Some`, each
    /// entry replaces the element-derived radius for that atom in the built-in
    /// ball / impostor rendering; `None` keeps the element defaults.
    pub atom_radii: Option<Vec<f32>>,
    /// Optional per-atom RGBA color override (atom order). When `Some`, each
    /// entry replaces `color_fn`'s result for that atom; `None` uses `color_fn`.
    pub atom_colors: Option<Vec<[f32; 4]>>,
    /// Optional per-atom visibility mask (atom order). See
    /// [`MoleculeViewer::set_visible_atoms`].
    visible_atoms: Option<Vec<bool>>,
    /// Number of `true` entries in `visible_atoms`, cached because the renderer
    /// consults it per frame to decide whether the mesh styles still fit.
    visible_count: Option<usize>,
    /// Periodic images the molecule is replicated into. See
    /// [`MoleculeViewer::set_periodic_images`].
    periodic_images: Option<PeriodicImages>,
    /// Spatial index for ray picking, built lazily and rebuilt when `revision`
    /// changes. Behind a `Mutex` because `pick` takes `&self` — building it
    /// there rather than eagerly in `set_molecule` keeps loading a molecule you
    /// never click on free.
    pick_grid: Mutex<Option<(u64, Arc<SpatialGrid>)>>,
}

impl Default for MoleculeViewer {
    fn default() -> Self {
        Self::new()
    }
}

impl MoleculeViewer {
    pub fn new() -> Self {
        Self {
            molecule: None,
            revision: 0,
            additional_render: Vec::new(),
            color_fn: default_color_fn,
            molecule_opacity: 1.0,
            atom_radii: None,
            atom_colors: None,
            visible_atoms: None,
            visible_count: None,
            periodic_images: None,
            pick_grid: Mutex::new(None),
        }
    }

    /// Create a new MoleculeViewer with a custom color function
    pub fn with_color_fn(color_fn: ColorFn) -> Self {
        Self {
            molecule: None,
            revision: 0,
            additional_render: Vec::new(),
            color_fn,
            molecule_opacity: 1.0,
            atom_radii: None,
            atom_colors: None,
            visible_atoms: None,
            visible_count: None,
            periodic_images: None,
            pick_grid: Mutex::new(None),
        }
    }

    /// A counter that changes whenever anything the built-in molecule geometry
    /// is built from changes: the molecule itself, its positions, the color
    /// function, the whole-molecule opacity, or the per-atom radius / color
    /// overrides.
    ///
    /// The renderer keys its geometry cache on this, so it is the single answer
    /// to "does the cached geometry still describe this viewer?". Callers
    /// mutating the molecule through [`Self::molecule`] directly — the field is
    /// public — must call [`Self::mark_changed`] afterwards, or the renderer
    /// will keep drawing the previous geometry.
    #[inline]
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Declare that the molecule was mutated behind the viewer's back, through
    /// the public [`Self::molecule`] field.
    #[inline]
    pub fn mark_changed(&mut self) {
        self.bump_revision();
    }

    #[inline]
    fn bump_revision(&mut self) {
        self.revision = self.revision.wrapping_add(1);
    }

    /// Set the color function
    pub fn set_color_fn(&mut self, color_fn: ColorFn) {
        self.color_fn = color_fn;
        self.bump_revision();
    }

    /// Set the whole-molecule opacity (clamped to `0.0..=1.0`). `1.0` is opaque.
    pub fn set_molecule_opacity(&mut self, opacity: f32) {
        self.molecule_opacity = opacity.clamp(0.0, 1.0);
        self.bump_revision();
    }

    /// Set (or clear) the per-atom radius override used by the built-in molecule
    /// rendering. Pass `None` to fall back to element-derived radii.
    pub fn set_atom_radii(&mut self, radii: Option<Vec<f32>>) {
        self.atom_radii = radii;
        self.bump_revision();
    }

    /// Set (or clear) the per-atom RGBA color override used by the built-in
    /// molecule rendering. Pass `None` to fall back to `color_fn`.
    pub fn set_atom_colors(&mut self, colors: Option<Vec<[f32; 4]>>) {
        self.atom_colors = colors;
        self.bump_revision();
    }

    /// Hide part of the molecule without rebuilding it.
    ///
    /// `None` shows everything (the default, and allocation-free). With a mask,
    /// `mask[i] == false` hides atom `i` and every bond incident to it.
    ///
    /// **Indices never renumber.** Picking results, [`Self::set_atom_radii`],
    /// [`Self::set_atom_colors`], [`Self::update_positions`] and every
    /// `AdditionalRender` continue to address the full molecule, so a caller
    /// showing a subset does not have to maintain a parallel index space —
    /// which is the whole point of doing this here rather than by pushing a
    /// filtered copy of the molecule.
    ///
    /// A mask shorter than the molecule leaves the remaining atoms visible.
    pub fn set_visible_atoms(&mut self, visible: Option<Vec<bool>>) {
        self.visible_count = visible
            .as_ref()
            .map(|mask| mask.iter().filter(|shown| **shown).count());
        self.visible_atoms = visible;
        self.bump_revision();
    }

    /// The current visibility mask, or `None` when everything is visible.
    pub fn visible_atoms(&self) -> Option<&[bool]> {
        self.visible_atoms.as_deref()
    }

    /// How many atoms are currently drawn.
    pub fn visible_atom_count(&self) -> usize {
        let total = self.molecule.as_ref().map(|m| m.atoms.len()).unwrap_or(0);
        match (self.visible_count, self.visible_atoms.as_ref()) {
            // A mask shorter than the molecule leaves the rest visible.
            (Some(counted), Some(mask)) => counted + total.saturating_sub(mask.len()),
            _ => total,
        }
    }

    /// Whether atom `index` is currently drawn.
    #[inline]
    pub fn is_atom_visible(&self, index: usize) -> bool {
        crate::frame_state::is_visible(self.visible_atoms.as_deref(), index)
    }

    /// Draw the molecule in periodic images of a simulation cell as well as in
    /// the primary cell.
    ///
    /// The geometry is uploaded once and drawn once per image with a different
    /// translation, so replicating costs draw calls but no extra memory and no
    /// CPU rebuild — which is what makes it usable on systems where duplicating
    /// the atoms 27 times would not be.
    ///
    /// Picking sees the images too: clicking a replica selects the atom it is a
    /// copy of, in primary-cell indices.
    pub fn set_periodic_images(&mut self, images: Option<PeriodicImages>) {
        self.periodic_images = images.filter(|images| !images.is_trivial());
        // Only the draw translation changes, not the geometry -- but picking
        // caches its answer against this counter, so it has to move.
        self.bump_revision();
    }

    /// The periodic images currently drawn, or `None` for the primary cell only.
    pub fn periodic_images(&self) -> Option<&PeriodicImages> {
        self.periodic_images.as_ref()
    }

    pub fn set_molecule(&mut self, molecule: Molecule) {
        self.molecule = Some(molecule);
        self.bump_revision();
    }

    /// Update the loaded molecule's atom positions in place for trajectory
    /// playback. Elements, bonds and metadata are untouched, so feeding
    /// successive frames reuses all existing storage. `positions` must match
    /// the atom count and be in the crate's nanometer units. Returns `Err` if
    /// no molecule is loaded or the count mismatches.
    pub fn update_positions(&mut self, positions: &[Vec3]) -> Result<(), String> {
        let mol = self
            .molecule
            .as_mut()
            .ok_or_else(|| "no molecule loaded".to_string())?;
        mol.set_positions(positions)?;
        self.bump_revision();
        Ok(())
    }

    /// Like [`update_positions`](Self::update_positions) but takes Ångström
    /// coordinates, applying the crate's Å→nm conversion.
    pub fn update_positions_angstrom(&mut self, coords: &[[f32; 3]]) -> Result<(), String> {
        let mol = self
            .molecule
            .as_mut()
            .ok_or_else(|| "no molecule loaded".to_string())?;
        mol.set_positions_angstrom(coords)?;
        self.bump_revision();
        Ok(())
    }

    pub fn add_additional_render<R: AdditionalRender + 'static>(&mut self, render: R) {
        self.additional_render.push(Box::new(render));
        self.bump_revision();
    }

    /// Add a boxed `AdditionalRender` directly.
    pub fn add_additional_render_box(&mut self, render: Box<dyn AdditionalRender>) {
        self.additional_render.push(render);
        self.bump_revision();
    }

    /// Ray-test the molecule and report the nearest atom or bond hit.
    ///
    /// `ray_dir` must be normalised (which is what
    /// [`crate::Camera::ray_from_screen`] returns), so the intersection
    /// parameter is a distance.
    pub fn pick(&self, ray_origin: Vec3, ray_dir: Vec3) -> Option<ViewerEvent> {
        let mut closest_t = f32::MAX;
        let mut picked = None;

        // Testing the ray against a molecule translated into an image is the
        // same as testing the *translated ray* against the molecule where it
        // is, so every image reuses the one spatial grid. The direction is
        // unchanged, so the hit distances stay directly comparable and the
        // nearest hit across all images wins -- which is what the user sees.
        match self.periodic_images.as_ref() {
            Some(images) => {
                for translation in images.translations() {
                    self.pick_in_image(
                        ray_origin - translation,
                        ray_dir,
                        &mut closest_t,
                        &mut picked,
                    );
                }
            }
            None => self.pick_in_image(ray_origin, ray_dir, &mut closest_t, &mut picked),
        }

        Some(picked.unwrap_or(ViewerEvent::NothingClicked))
    }

    /// Ray-test the primary cell, keeping `closest_t`/`picked` if nothing here
    /// beats what is already there.
    ///
    /// Reported indices are always primary-cell indices, so clicking a periodic
    /// replica selects the atom it is a copy of.
    fn pick_in_image(
        &self,
        ray_origin: Vec3,
        ray_dir: Vec3,
        closest_t: &mut f32,
        picked: &mut Option<ViewerEvent>,
    ) {
        {
            let Some(mol) = &self.molecule else {
                return;
            };
            // Atoms, through the spatial grid: walk the cells the ray actually
            // crosses, nearest first, and stop as soon as no unvisited cell can
            // contain anything closer. A linear scan here cost one ray-sphere
            // test per atom per pointer move -- millions, on an MD system.
            let grid = self.atom_pick_grid();
            grid.for_each_along_ray(ray_origin, ray_dir, |candidates, cell_entry_t| {
                // Cells are visited in increasing distance, so nothing beyond
                // this one can beat a hit we already have.
                if *closest_t < cell_entry_t {
                    return false;
                }

                for &candidate in candidates {
                    let i = candidate as usize;
                    // Hidden atoms are not drawn, so they must not be clickable
                    // either -- otherwise clicking through a hidden shell
                    // selects something the user cannot see.
                    if !self.is_atom_visible(i) {
                        continue;
                    }
                    let Some(atom) = mol.atoms.get(i) else {
                        continue;
                    };
                    let radius = ball_stick_radius(&atom.element, false);
                    if let Some(t) =
                        Self::ray_sphere_intersect(ray_origin, ray_dir, atom.position, radius)
                    {
                        if t < *closest_t && t > 0.0 {
                            *closest_t = t;
                            *picked = Some(ViewerEvent::AtomClicked(i));
                        }
                    }
                }
                true
            });

            // Bonds. The radius must match what the ball-and-stick style draws
            // its sticks at (`ball_stick_style::build_vertices`, which uses the
            // same `default_ball_stick_bond_radius()`), or bonds become
            // clickable somewhere other than where they appear.
            //
            // Left as a linear scan: bonds are not in the grid, and a molecule
            // with bonds is one whose connectivity came from a file, which caps
            // it well below the atom counts that made the atom scan a problem.
            // The enclosing-sphere prefilter below rejects most of them for
            // about a quarter of the cost of the full cylinder test.
            let radius = default_ball_stick_bond_radius();
            for (i, bond) in mol.bonds.iter().enumerate() {
                if !self.is_atom_visible(bond.atom_a) || !self.is_atom_visible(bond.atom_b) {
                    continue;
                }
                let Some((p1, p2)) = mol.bond_endpoints(bond) else {
                    continue;
                };

                let mid = (p1 + p2) * 0.5;
                let enclosing = (p2 - p1).magnitude() * 0.5 + radius;
                if !Self::ray_hits_sphere(ray_origin, ray_dir, mid, enclosing) {
                    continue;
                }

                if let Some(t) = Self::ray_cylinder_intersect(ray_origin, ray_dir, p1, p2, radius) {
                    if t < *closest_t && t > 0.0 {
                        *closest_t = t;
                        *picked = Some(ViewerEvent::BondClicked(i));
                    }
                }
            }
        }
    }

    /// The atom grid for picking, built on first use and reused until the
    /// geometry changes.
    fn atom_pick_grid(&self) -> Arc<SpatialGrid> {
        let revision = self.revision;
        if let Ok(cache) = self.pick_grid.lock() {
            if let Some((cached_revision, grid)) = cache.as_ref() {
                if *cached_revision == revision {
                    return Arc::clone(grid);
                }
            }
        }

        let atoms: &[Atom] = self
            .molecule
            .as_ref()
            .map(|mol| mol.atoms.as_slice())
            .unwrap_or(&[]);
        let radii: Vec<f32> = atoms
            .iter()
            .map(|atom| ball_stick_radius(&atom.element, false))
            .collect();
        // A cell no smaller than the largest sphere keeps the per-atom cell
        // fan-out at build time down to a handful.
        let cell_size = radii.iter().fold(0.0_f32, |m, &r| m.max(r)) * 4.0;
        let grid = Arc::new(SpatialGrid::spheres(atoms, cell_size.max(1e-3), |i| {
            radii[i]
        }));

        if let Ok(mut cache) = self.pick_grid.lock() {
            *cache = Some((revision, Arc::clone(&grid)));
        }
        grid
    }

    /// Whether the ray passes within `radius` of `center` at a non-negative
    /// distance. Cheaper than a full intersection when only rejection matters.
    #[inline]
    fn ray_hits_sphere(ray_origin: Vec3, ray_dir: Vec3, center: Vec3, radius: f32) -> bool {
        let to_center = center - ray_origin;
        let along = to_center.dot(ray_dir);
        // Behind the ray and further than its own radius: cannot be hit.
        if along < -radius {
            return false;
        }
        let perpendicular_sq = to_center.magnitude_squared() - along * along;
        perpendicular_sq <= radius * radius
    }

    fn ray_sphere_intersect(
        ray_origin: Vec3,
        ray_dir: Vec3,
        center: Vec3,
        radius: f32,
    ) -> Option<f32> {
        let l = center - ray_origin;
        let tca = l.dot(ray_dir);
        if tca < 0.0 {
            return None;
        }
        let d2 = l.dot(l) - tca * tca;
        let r2 = radius * radius;
        if d2 > r2 {
            return None;
        }
        let thc = (r2 - d2).sqrt();
        Some(tca - thc)
    }

    fn ray_cylinder_intersect(
        ray_origin: Vec3,
        ray_dir: Vec3,
        p1: Vec3,
        p2: Vec3,
        radius: f32,
    ) -> Option<f32> {
        let ba = p2 - p1;
        let oa = ray_origin - p1;
        let baba = ba.dot(ba);
        let bard = ba.dot(ray_dir);
        let baoa = ba.dot(oa);
        let roa = oa.dot(ray_dir);
        let oaoa = oa.dot(oa);

        let a = baba - bard * bard;
        let b = baba * roa - baoa * bard;
        let c = baba * oaoa - baoa * baoa - radius * radius * baba;
        let h = b * b - a * c;

        if h >= 0.0 {
            let t = (-b - h.sqrt()) / a;
            let y = baoa + t * bard;
            // Check body
            if y > 0.0 && y < baba {
                return Some(t);
            }
            // Caps are not checked here for simplicity, but usually fine for picking
        }
        None
    }

}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::{Bond, Element};
    use crate::overlays::SimulationCellState;

    fn atom_at(x: f32, y: f32, z: f32, id: usize) -> Atom {
        Atom {
            position: Vec3::new(x, y, z),
            element: Element::new("C"),
            id,
            meta: None,
        }
    }

    /// A single atom in a 10 nm cubic cell.
    fn one_atom_viewer() -> MoleculeViewer {
        let mut viewer = MoleculeViewer::new();
        viewer.set_molecule(Molecule::from_atoms_bonds(
            vec![atom_at(0.0, 0.0, 0.0, 0)],
            Vec::new(),
        ));
        viewer
    }

    fn cubic_cell(edge: f32) -> SimulationCellState {
        SimulationCellState::rectangular(Vec3::new(edge, edge, edge))
    }

    /// Aim at where the +x replica sits. Without periodic images there is
    /// nothing there; with them, the replica is hit and reports the *primary*
    /// atom, because that is the atom the user means.
    #[test]
    fn picking_reaches_periodic_replicas_and_reports_primary_indices() {
        let mut viewer = one_atom_viewer();
        let replica = Vec3::new(10.0, 0.0, 0.0);

        // Fire down -z at the replica's column.
        let origin = replica + Vec3::new(0.0, 0.0, 5.0);
        let direction = Vec3::new(0.0, 0.0, -1.0);

        assert!(
            matches!(
                viewer.pick(origin, direction),
                Some(ViewerEvent::NothingClicked)
            ),
            "nothing is there until the cell is replicated"
        );

        viewer.set_periodic_images(Some(PeriodicImages::new(cubic_cell(10.0), [1, 0, 0])));
        assert!(
            matches!(
                viewer.pick(origin, direction),
                Some(ViewerEvent::AtomClicked(0))
            ),
            "the replica should pick the atom it is a copy of"
        );
    }

    /// Two replicas along the ray: the nearer one has to win, or clicking
    /// selects something behind what the pointer is over.
    #[test]
    fn the_nearest_image_wins() {
        let mut viewer = MoleculeViewer::new();
        viewer.set_molecule(Molecule::from_atoms_bonds(
            vec![atom_at(0.0, 0.0, 0.0, 0), atom_at(0.0, 0.0, 3.0, 1)],
            Vec::new(),
        ));
        viewer.set_periodic_images(Some(PeriodicImages::new(cubic_cell(10.0), [0, 0, 1])));

        // Looking down -z from far away: atom 1 (z = 3) is nearest.
        let picked = viewer.pick(Vec3::new(0.0, 0.0, 40.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(matches!(picked, Some(ViewerEvent::AtomClicked(1))), "{picked:?}");
    }

    #[test]
    fn hidden_atoms_stay_unclickable_in_every_image() {
        let mut viewer = one_atom_viewer();
        viewer.set_periodic_images(Some(PeriodicImages::new(cubic_cell(10.0), [1, 0, 0])));
        viewer.set_visible_atoms(Some(vec![false]));

        let origin = Vec3::new(10.0, 0.0, 5.0);
        assert!(matches!(
            viewer.pick(origin, Vec3::new(0.0, 0.0, -1.0)),
            Some(ViewerEvent::NothingClicked)
        ));
    }

    /// Replication only changes where geometry is drawn, but picking caches its
    /// answer against the revision, so the revision has to move with it.
    #[test]
    fn changing_the_images_bumps_the_revision() {
        let mut viewer = one_atom_viewer();
        let before = viewer.revision();

        viewer.set_periodic_images(Some(PeriodicImages::new(cubic_cell(10.0), [1, 1, 1])));
        assert_ne!(before, viewer.revision());
        assert!(viewer.periodic_images().is_some());

        // A replication that draws nothing extra is not worth carrying.
        viewer.set_periodic_images(Some(PeriodicImages::new(cubic_cell(10.0), [0, 0, 0])));
        assert!(viewer.periodic_images().is_none());
    }

    /// Bonds are replicated with their atoms, so a replica's stick is clickable
    /// too -- and reports the primary bond.
    #[test]
    fn bond_picking_reaches_replicas() {
        let mut viewer = MoleculeViewer::new();
        viewer.set_molecule(Molecule::from_atoms_bonds(
            vec![atom_at(-1.0, 0.0, 0.0, 0), atom_at(1.0, 0.0, 0.0, 1)],
            vec![Bond {
                atom_a: 0,
                atom_b: 1,
                order: 1,
            }],
        ));
        viewer.set_periodic_images(Some(PeriodicImages::new(cubic_cell(10.0), [0, 1, 0])));

        // Aim at the midpoint of the +y replica's bond, which lies between its
        // two atoms and so can only be a bond hit.
        let picked = viewer.pick(Vec3::new(0.0, 10.0, 5.0), Vec3::new(0.0, 0.0, -1.0));
        assert!(matches!(picked, Some(ViewerEvent::BondClicked(0))), "{picked:?}");
    }
}
