use crate::molecule::Molecule;
use graphics::{Entity, Mesh, Scene};
use lin_alg::f32::{Quaternion, Vec3};

#[derive(Debug, Clone)]
pub enum ViewerEvent {
    AtomClicked(usize),
    BondClicked(usize),
    NothingClicked,
}

pub struct MoleculeViewer {
    pub molecule: Option<Molecule>,
    pub dirty: bool,
    pub last_mouse_pos: (f32, f32),
}

impl MoleculeViewer {
    pub fn new() -> Self {
        Self {
            molecule: None,
            dirty: false,
            last_mouse_pos: (0.0, 0.0),
        }
    }

    pub fn set_molecule(&mut self, molecule: Molecule) {
        self.molecule = Some(molecule);
        self.dirty = true;
    }

    pub fn on_mouse_move(&mut self, pos: (f32, f32)) {
        self.last_mouse_pos = pos;
    }

    pub fn pick(&self, ray_origin: Vec3, ray_dir: Vec3) -> Option<ViewerEvent> {
        let mut closest_t = f32::MAX;
        let mut picked = None;

        if let Some(mol) = &self.molecule {
            // Check Atoms
            for (i, atom) in mol.atoms.iter().enumerate() {
                let pos = Vec3::new(atom.position.x, atom.position.y, atom.position.z);
                let radius = 0.4; // Must match update_scene
                if let Some(t) = Self::ray_sphere_intersect(ray_origin, ray_dir, pos, radius) {
                    if t < closest_t && t > 0.0 {
                        closest_t = t;
                        picked = Some(ViewerEvent::AtomClicked(i));
                    }
                }
            }

            // Check Bonds
            for (i, bond) in mol.bonds.iter().enumerate() {
                let a = mol.atoms[bond.atom_a].position;
                let b = mol.atoms[bond.atom_b].position;
                let p1 = Vec3::new(a.x, a.y, a.z);
                let p2 = Vec3::new(b.x, b.y, b.z);
                let radius = 0.15; // Must match update_scene

                if let Some(t) = Self::ray_cylinder_intersect(ray_origin, ray_dir, p1, p2, radius) {
                    if t < closest_t && t > 0.0 {
                        closest_t = t;
                        picked = Some(ViewerEvent::BondClicked(i));
                    }
                }
            }
        }

        picked.or(Some(ViewerEvent::NothingClicked))
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
    pub fn update_scene(&mut self, scene: &mut Scene) {
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
                // Convert nalgebra Point3 to graphics Vec3
                // Assuming nalgebra::Point3 fields are x, y, z or coords[0], etc.
                // But atom.position is Point3 from nalgebra.
                let pos = Vec3::new(atom.position.x, atom.position.y, atom.position.z);

                let color = match atom.element.as_str() {
                    "C" => (0.1, 0.1, 0.1),  // Black/Dark Grey
                    "H" => (0.9, 0.9, 0.9),  // White
                    "O" => (0.9, 0.1, 0.1),  // Red
                    "N" => (0.1, 0.1, 0.9),  // Blue
                    "S" => (0.9, 0.9, 0.1),  // Yellow
                    "P" => (1.0, 0.6, 0.0),  // Orange
                    "Cl" => (0.1, 0.9, 0.1), // Green
                    _ => (0.7, 0.7, 0.7),    // Grey
                };

                let radius = 0.4; // Base radius

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

                let bond_radius = 0.15;
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
        }
    }
}
