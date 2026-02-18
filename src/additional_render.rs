use crate::molecule::Molecule;
use graphics::{Entity, Mesh, Scene};
use lin_alg::f32::Quaternion;
use lin_alg::f32::Vec3;


// for adding rendering works to MoleculeViewer.
pub trait AdditionalRender {
    fn update_scene(&self, scene: &mut Scene, molecule: &Molecule);
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
   
}
    
impl DebugRender {
    pub fn new(ray: (Vec3, Vec3)) -> Self {
        Self { ray }
    }
}

impl AdditionalRender for DebugRender {
    fn update_scene(&self, _scene: &mut Scene, _molecule: &Molecule) {
        // For debugging purposes, we can add some simple geometry or text here.
        // For example, we could render the coordinate axes or display some text info.
        
        // draw ray
        let (origin, direction) = self.ray;
        let ray_mesh = Mesh::new_cylinder(0.05, 1.0, 10);
        let ray_idx = _scene.meshes.len();
        _scene.meshes.push(ray_mesh);
        _scene.entities.push(Entity::new(
            ray_idx,
            origin,
            Quaternion::new_identity(),
            1.0,
            (0.0, 1.0, 0.0),
            0.2,
        ));

    }
}

impl DebugRender {
    pub fn update_ray(&mut self, ray: (Vec3, Vec3)) {
        self.ray = ray;
    }
}   