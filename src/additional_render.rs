use crate::molecule::Molecule;
use crate::render_state::{get_state_clone_by_type, SharedRenderStates};
use crate::scene_types::{Entity, Mesh, Scene};
use lin_alg::f32::Quaternion;
use lin_alg::f32::Vec3;

// for adding rendering works to MoleculeViewer.
pub trait AdditionalRender: Send {
    /// Update the given `scene` using `molecule` and the shared `states` map.
    fn update_scene(&self, scene: &mut Scene, molecule: &Molecule, states: &SharedRenderStates);

    fn add_sphere(&self, scene: &mut Scene, position: Vec3, radius: f32, color: (f32, f32, f32)) {
        let sphere_mesh = Mesh::new_sphere(1.0, 16);
        let mesh_idx = scene.meshes.len();
        scene.meshes.push(sphere_mesh);

        let entity = Entity::new(
            mesh_idx,
            position,
            Quaternion::new_identity(),
            radius,
            color,
            0.2,
        );
        scene.entities.push(entity);
    }

    fn add_cylinder(
        &self,
        scene: &mut Scene,
        start: Vec3,
        end: Vec3,
        radius: f32,
        color: (f32, f32, f32),
    ) {
        let cyl_mesh = Mesh::new_cylinder(1.0, 1.0, 10);
        let mesh_idx = scene.meshes.len();
        scene.meshes.push(cyl_mesh);

        let mid_point = (start + end) * 0.5;
        let direction = end - start;
        let length = direction.magnitude();
        let orientation =
            Quaternion::from_unit_vecs(Vec3::new(0.0, 1.0, 0.0), direction.to_normalized());

        let mut entity = Entity::new(mesh_idx, mid_point, orientation, 1.0, color, 0.2);
        entity.scale_partial = Some(Vec3::new(radius, length, radius));
        scene.entities.push(entity);
    }
}
#[derive(Clone)]
pub struct SelectedAtomRender {
    pub selected_atoms: Vec<usize>,
    pub color: [f32; 3],
}

impl SelectedAtomRender {
    pub fn new() -> Self {
        Self {
            selected_atoms: Vec::new(),
            color: [1.0, 0.0, 0.0],
        }
    }

    pub fn selected_atoms(&self) -> &[usize] {
        &self.selected_atoms
    }

    pub fn set_selected_atoms(&mut self, atom_indices: Vec<usize>) {
        self.selected_atoms = atom_indices;
    }

    pub fn clear_selected_atoms(&mut self) {
        self.selected_atoms.clear();
    }
}

impl AdditionalRender for SelectedAtomRender {
    fn update_scene(&self, scene: &mut Scene, molecule: &Molecule, states: &SharedRenderStates) {
        // Use type-keyed state to lookup selected atoms. If not set, fall back to internal list.
        let source: Vec<usize> = get_state_clone_by_type::<Vec<usize>>(states)
            .unwrap_or_else(|| self.selected_atoms.clone());

        if source.is_empty() {
            return;
        }

        let radius = 0.6;
        let color = (self.color[0], self.color[1], self.color[2]);
        scene.entities.reserve(source.len());

        for atom_idx in source.iter() {
            if let Some(atom) = molecule.atoms.get(*atom_idx) {
                self.add_sphere(scene, atom.position, radius, color);
            }
        }
    }
}

impl SelectedAtomRender {
    pub fn add_atom(&mut self, atom_idx: usize) {
        if !self.selected_atoms.contains(&atom_idx) {
            self.selected_atoms.push(atom_idx);
        }
    }

    pub fn remove_atom(&mut self, atom_idx: usize) {
        if let Some(pos) = self.selected_atoms.iter().position(|&x| x == atom_idx) {
            self.selected_atoms.remove(pos);
        }
    }

    pub fn toggle_atom(&mut self, atom_idx: usize) {
        if let Some(pos) = self.selected_atoms.iter().position(|&x| x == atom_idx) {
            self.selected_atoms.remove(pos);
        } else {
            self.selected_atoms.push(atom_idx);
        }
    }
}

pub struct DebugRender {
    pub ray: (Vec3, Vec3),
    pub ray_length: f32,
    pub ray_color: (f32, f32, f32),
}

impl DebugRender {
    pub fn new(ray: (Vec3, Vec3)) -> Self {
        Self {
            ray,
            ray_length: 100.0,
            ray_color: (0.0, 1.0, 0.0), // Default green color
        }
    }

    /// Create a debug renderer with custom ray length
    pub fn with_length(ray: (Vec3, Vec3), length: f32) -> Self {
        Self {
            ray,
            ray_length: length,
            ray_color: (0.0, 1.0, 0.0),
        }
    }

    /// Create a debug renderer with custom color
    pub fn with_color(ray: (Vec3, Vec3), color: (f32, f32, f32)) -> Self {
        Self {
            ray,
            ray_length: 100.0,
            ray_color: color,
        }
    }

    pub fn set_ray_length(&mut self, length: f32) {
        self.ray_length = length;
    }

    pub fn set_ray_color(&mut self, color: (f32, f32, f32)) {
        self.ray_color = color;
    }
}

impl AdditionalRender for DebugRender {
    fn update_scene(&self, scene: &mut Scene, _molecule: &Molecule, _states: &SharedRenderStates) {
        // Draw debug ray as a thin cylinder
        let (origin, direction) = self.ray;

        // Normalize direction
        let normalized_dir = direction.to_normalized();
        let ray_radius = 0.05; // Thin cylinder for visualization

        // Calculate midpoint of the ray
        let ray_end = origin + normalized_dir * self.ray_length;
        let midpoint = (origin + ray_end) * 0.5;

        // Create cylinder mesh
        let ray_mesh = Mesh::new_cylinder(1.0, 1.0, 8);
        let ray_idx = scene.meshes.len();
        scene.meshes.push(ray_mesh);

        // Quaternion to rotate from Y-axis (default cylinder orientation) to ray direction
        let up = Vec3::new(0.0, 1.0, 0.0);
        let orientation = Quaternion::from_unit_vecs(up, normalized_dir);

        // Create entity with proper scaling
        let mut ray_entity = Entity::new(ray_idx, midpoint, orientation, 1.0, self.ray_color, 0.1);

        // Apply scale_partial to set cylinder dimensions
        // X and Z are radii, Y is length
        ray_entity.scale_partial = Some(Vec3::new(ray_radius, self.ray_length, ray_radius));
        scene.entities.push(ray_entity);
    }
}

impl DebugRender {
    pub fn update_ray(&mut self, ray: (Vec3, Vec3)) {
        self.ray = ray;
    }
}
