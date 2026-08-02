//! Highlight for the currently selected atoms.

use crate::atom_radii::vdw_radius;
use crate::frame_state::RenderFrameState;
use crate::render_state::with_state_by_type;
use crate::scene_types::Scene;
use crate::AdditionalRender;

#[derive(Clone)]
pub struct SelectedAtomRenderState {
    pub selected_atoms: Vec<usize>,
    /// RGBA highlight color; the alpha component controls transparency.
    pub color: [f32; 4],
}
pub struct SelectedAtomRender {}

impl SelectedAtomRender {
    pub fn new() -> Self {
        Self {}
    }
}

impl AdditionalRender for SelectedAtomRender {
    fn update_scene(&self, scene: &mut Scene, frame: &RenderFrameState<'_>) {
        let Some(molecule) = frame.molecule else {
            return;
        };

        let Some(states) = frame.shared_states else {
            return;
        };

        // Borrow the selection rather than cloning it: this runs once per frame
        // and the list is as long as the user's selection.
        with_state_by_type::<SelectedAtomRenderState, ()>(states, |source| {
            let color = (
                source.color[0],
                source.color[1],
                source.color[2],
                source.color[3],
            );
            scene.entities.reserve(source.selected_atoms.len());
            for atom_idx in &source.selected_atoms {
                if let Some(atom) = molecule.atoms.get(*atom_idx) {
                    self.add_sphere(
                        scene,
                        frame,
                        atom.position,
                        vdw_radius(&atom.element) * 0.4,
                        color,
                    );
                }
            }
        });
    }
}

impl SelectedAtomRenderState {
    pub fn new(selected_atoms: Vec<usize>, color: [f32; 4]) -> Self {
        Self {
            selected_atoms,
            color,
        }
    }

    pub fn set_color(&mut self, color: [f32; 4]) {
        self.color = color;
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
