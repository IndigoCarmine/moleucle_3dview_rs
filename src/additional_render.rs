use std::cell::RefCell;

use crate::molecule::Molecule;
use crate::viewer::ViewerEvent;
use graphics::{Entity, Mesh, Scene};
use lin_alg::f32::Quaternion;
use lin_alg::f32::Vec3;

pub trait AdditionalRender {
    fn update_scene(&self, scene: &mut Scene, molecule: &Molecule);
    fn handle_event(&mut self, _event: &ViewerEvent) {}
}

pub struct SelectedAtomRender {
    pub selected_atoms: RefCell<Vec<usize>>,
    pub color: [f32; 3],
}

impl SelectedAtomRender {
    pub fn new() -> Self {
        Self {
            selected_atoms: RefCell::new(Vec::new()),
            color: [1.0, 0.0, 0.0],
        }
    }
}

impl AdditionalRender for SelectedAtomRender {
    fn update_scene(&self, scene: &mut Scene, molecule: &Molecule) {
        for atom_idx in self.selected_atoms.borrow().iter() {
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

    fn handle_event(&mut self, event: &ViewerEvent) {
        if let ViewerEvent::AtomClicked(idx) = event {
            self.toggle_atom(*idx);
        }
    }
}

impl SelectedAtomRender {
    pub fn add_atom(&mut self, atom_idx: usize) {
        self.selected_atoms.borrow_mut().push(atom_idx);
    }

    pub fn remove_atom(&mut self, atom_idx: usize) {
        self.selected_atoms.borrow_mut().retain(|&x| x != atom_idx);
    }

    pub fn toggle_atom(&mut self, atom_idx: usize) {
        if self.selected_atoms.borrow().contains(&atom_idx) {
            self.remove_atom(atom_idx);
        } else {
            self.add_atom(atom_idx);
        }
    }
}
