use crate::molecule::Molecule;
use crate::scene_types::{Entity, Mesh, Scene};
use lin_alg::f32::Quaternion;
use lin_alg::f32::Vec3;
use std::any::Any;


// for adding rendering works to MoleculeViewer.
pub trait AdditionalRender: Send {
    fn update_scene(&self, scene: &mut Scene, molecule: &Molecule);
    
    fn as_any_mut(&mut self) -> &mut dyn Any;
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
    fn update_scene(&self, scene: &mut Scene, molecule: &Molecule) {
        for atom_idx in self.selected_atoms.iter() {
            let atom = molecule.atoms.get(*atom_idx).unwrap();
            let pos = Vec3::new(atom.position.x, atom.position.y, atom.position.z);
            let radius = 0.4 + 0.2;
            let color = self.color;
            let cyl_mesh = Mesh::new_cylinder(1.0, 1.0, 10);
            let cyl_idx = scene.meshes.len();
            scene.meshes.push(cyl_mesh);
            scene.entities.push(Entity::new(
                cyl_idx,
                pos,
                Quaternion::new_identity(),
                radius,
                (color[0], color[1], color[2]),
                0.2,
            ));
        }
    }
    
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}


impl SelectedAtomRender {
    pub fn add_atom(&mut self, atom_idx: usize) {
        self.selected_atoms.push(atom_idx);
    }

    pub fn remove_atom(&mut self, atom_idx: usize) {
        self.selected_atoms.retain(|&x| x != atom_idx);
    }

    pub fn toggle_atom(&mut self, atom_idx: usize) {
        if self.selected_atoms.contains(&atom_idx) {
            self.remove_atom(atom_idx);
        } else {
            self.add_atom(atom_idx);
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
    fn update_scene(&self, scene: &mut Scene, _molecule: &Molecule) {
        // Draw debug ray as a thin cylinder
        let (origin, direction) = self.ray;
        
        // Normalize direction
        let normalized_dir = direction.to_normalized();
        let ray_radius = 0.05;  // Thin cylinder for visualization
        
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
        let mut ray_entity = Entity::new(
            ray_idx,
            midpoint,
            orientation,
            1.0,
            self.ray_color,
            0.1,
        );
        
        // Apply scale_partial to set cylinder dimensions
        // X and Z are radii, Y is length
        ray_entity.scale_partial = Some(Vec3::new(ray_radius, self.ray_length, ray_radius));
        scene.entities.push(ray_entity);
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl DebugRender {
    pub fn update_ray(&mut self, ray: (Vec3, Vec3)) {
        self.ray = ray;
    }
}   