use crate::atom_radii::{ball_stick_radius, default_ball_stick_bond_radius};
use crate::frame_state::RenderFrameState;
use crate::molecule::{Atom, Molecule};
use crate::scene_types::{Entity, Mesh, Scene};
use crate::AdditionalRender;
// Viewer no longer owns shared state; state is passed in by the caller (viewport or user).
use lin_alg::f32::{Quaternion, Vec3};

#[derive(Debug, Clone)]
pub enum ViewerEvent {
    AtomClicked(usize),
    BondClicked(usize),
    NothingClicked,
}

/// Color function type: takes Atom and is_selected flag, returns RGB color
pub type ColorFn = fn(&Atom, bool) -> (f32, f32, f32);

/// Default color function based on element type
pub fn default_color_fn(atom: &Atom, _is_selected: bool) -> (f32, f32, f32) {
    match atom.element.as_str() {
        "C" => (0.1, 0.1, 0.1),  // Black/Dark Grey
        "H" => (0.9, 0.9, 0.9),  // White
        "O" => (0.9, 0.1, 0.1),  // Red
        "N" => (0.1, 0.1, 0.9),  // Blue
        "S" => (0.9, 0.9, 0.1),  // Yellow
        "P" => (1.0, 0.6, 0.0),  // Orange
        "Cl" => (0.1, 0.9, 0.1), // Green
        _ => (0.7, 0.7, 0.7),    // Grey
    }
}

pub struct MoleculeViewer {
    pub molecule: Option<Molecule>,
    pub dirty: bool,
    pub additional_render: Vec<Box<dyn AdditionalRender>>,
    pub color_fn: ColorFn,
}

impl MoleculeViewer {
    pub fn new() -> Self {
        Self {
            molecule: None,
            dirty: false,
            additional_render: Vec::new(),
            color_fn: default_color_fn,
        }
    }

    /// Create a new MoleculeViewer with a custom color function
    pub fn with_color_fn(color_fn: ColorFn) -> Self {
        Self {
            molecule: None,
            dirty: false,
            additional_render: Vec::new(),
            color_fn,
        }
    }

    /// Set the color function
    pub fn set_color_fn(&mut self, color_fn: ColorFn) {
        self.color_fn = color_fn;
        self.dirty = true;
    }

    pub fn set_molecule(&mut self, molecule: Molecule) {
        self.molecule = Some(molecule);
        self.dirty = true;
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
        self.dirty = true;
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
        self.dirty = true;
        Ok(())
    }

    pub fn add_additional_render<R: AdditionalRender + 'static>(&mut self, render: R) {
        // keep the render's ownership and mark dirty
        self.additional_render.push(Box::new(render));
        self.dirty = true;
    }

    /// Add a boxed `AdditionalRender` directly.
    pub fn add_additional_render_box(&mut self, render: Box<dyn AdditionalRender>) {
        self.additional_render.push(render);
        self.dirty = true;
    }

    pub fn pick(&self, ray_origin: Vec3, ray_dir: Vec3) -> Option<ViewerEvent> {
        let mut closest_t = f32::MAX;
        let mut picked = None;

        if let Some(mol) = &self.molecule {
            // Check Atoms
            for (i, atom) in mol.atoms.iter().enumerate() {
                let radius = ball_stick_radius(&atom.element, false);
                if let Some(t) =
                    Self::ray_sphere_intersect(ray_origin, ray_dir, atom.position, radius)
                {
                    if t < closest_t && t > 0.0 {
                        closest_t = t;
                        picked = Some(ViewerEvent::AtomClicked(i));
                    }
                }
            }

            // Check Bonds
            for (i, bond) in mol.bonds.iter().enumerate() {
                let p1 = mol.atoms[bond.atom_a].position;
                let p2 = mol.atoms[bond.atom_b].position;
                let radius = default_ball_stick_bond_radius(); // Must match update_scene

                if let Some(t) = Self::ray_cylinder_intersect(ray_origin, ray_dir, p1, p2, radius) {
                    if t < closest_t && t > 0.0 {
                        closest_t = t;
                        picked = Some(ViewerEvent::BondClicked(i));
                    }
                }
            }
        }

        let result = picked.unwrap_or(ViewerEvent::NothingClicked);

        Some(result)
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

    /// Updates the graphics scene based on the current molecule data.
    ///
    /// `states` is a map of per-renderer states supplied by the caller (e.g., the viewport).
    pub fn update_scene(&mut self, scene: &mut Scene, frame: &RenderFrameState<'_>) {
        if !self.dirty {
            return;
        }
        self.dirty = false;

        if let Some(mol) = &self.molecule {
            scene.meshes.clear();
            scene.entities.clear();

            // 1. Create Meshes
            // Sphere for atoms (Radius 1.0, but we scale it)
            // 3 subdivisions gives a decent sphere.
            let sphere_mesh = Mesh::new_sphere(1.0, 3);
            let sphere_idx = scene.meshes.len();
            scene.meshes.push(sphere_mesh);

            // Cylinder for bonds (Length 1.0, Radius 1.0, along Y)
            // 10 sides is enough for thin bonds
            let cyl_mesh = Mesh::new_cylinder(1.0, 1.0, 10);
            let cyl_idx = scene.meshes.len();
            scene.meshes.push(cyl_mesh);

            // 2. Create Entities
            // Atoms
            for atom in &mol.atoms {
                // Convert atom position to graphics Vec3.
                let pos = Vec3::new(atom.position.x, atom.position.y, atom.position.z);

                // Use custom color function
                let color = (self.color_fn)(atom, false);

                let radius = ball_stick_radius(&atom.element, false);

                scene.entities.push(Entity::new(
                    sphere_idx,
                    pos,
                    Quaternion::new_identity(),
                    radius, // Uniform scale
                    color,
                    0.2, // Low shininess
                ));
            }

            // Bonds
            for bond in &mol.bonds {
                let a = mol.atoms[bond.atom_a].position;
                let b = mol.atoms[bond.atom_b].position;

                let p1 = Vec3::new(a.x, a.y, a.z);
                let p2 = Vec3::new(b.x, b.y, b.z);

                let diff = p2 - p1;
                let len = diff.magnitude();

                // If atoms are overlapping, skip bond
                if len < 0.001 {
                    continue;
                }

                let mid = (p1 + p2) * 0.5;

                // Orientation: Rotate Y-up cylinder to match `diff` direction
                let dir = diff.to_normalized();
                let up = Vec3::new(0.0, 1.0, 0.0);

                // Calculate rotation from UP to DIR
                // Quaternion from cross product?
                // Let's rely on standard way:
                // axis = cross(u, v)
                // angle = acos(dot(u, v))
                // but we need to handle parallel case.

                let orientation = Quaternion::from_unit_vecs(up, dir);

                let bond_radius = default_ball_stick_bond_radius();
                let scale_partial = Vec3::new(bond_radius, len, bond_radius);

                let mut entity = Entity::new(
                    cyl_idx,
                    mid,
                    orientation,
                    1.0,             // Base scale, overridden by partial
                    (0.5, 0.5, 0.5), // Grey bonds
                    0.1,
                );
                entity.scale_partial = Some(scale_partial);
                scene.entities.push(entity);
            }

            // draw xyz axes for debugging
            let axis_len = 2.0;
            let axis_radius = 0.05;

            // X axis Color Red
            let mut x_axis = Entity::new(
                cyl_idx,
                Vec3::new(axis_len / 2.0, 0.0, 0.0),
                Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), -std::f32::consts::FRAC_PI_2),
                1.0,
                (1.0, 0.0, 0.0),
                0.1,
            );
            x_axis.scale_partial = Some(Vec3::new(axis_radius, axis_len, axis_radius));
            scene.entities.push(x_axis);

            // Y axis Color Green
            let mut y_axis = Entity::new(
                cyl_idx,
                Vec3::new(0.0, axis_len / 2.0, 0.0),
                Quaternion::new_identity(),
                1.0,
                (0.0, 1.0, 0.0),
                0.1,
            );
            y_axis.scale_partial = Some(Vec3::new(axis_radius, axis_len, axis_radius));
            scene.entities.push(y_axis);

            // Z axis   Color Blue
            let mut z_axis = Entity::new(
                cyl_idx,
                Vec3::new(0.0, 0.0, axis_len / 2.0),
                Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), std::f32::consts::FRAC_PI_2),
                1.0,
                (0.0, 0.0, 1.0),
                0.1,
            );
            z_axis.scale_partial = Some(Vec3::new(axis_radius, axis_len, axis_radius));
            scene.entities.push(z_axis);

            for render in &self.additional_render {
                render.update_scene(scene, frame);
            }
        }
    }
}
