//! Links drawn between pairs of atoms.

use crate::frame_state::RenderFrameState;
use crate::render_state::with_state_by_type;
use crate::scene_types::Scene;
use crate::{vdw_radius, AdditionalRender};

/// Atom pairs to connect, as **0-based** indices into the rendered molecule.
///
/// Useful for anything the molecule's own bond list does not carry: declared
/// non-bonded interactions, restraints, hydrogen bonds, contacts. Pairs naming
/// an atom that does not exist are skipped.
#[derive(Clone, Debug, Default)]
pub struct AtomPairState {
    pub pairs: Vec<(usize, usize)>,
}

impl AtomPairState {
    pub fn new(pairs: Vec<(usize, usize)>) -> Self {
        Self { pairs }
    }
}

/// Draws a cylinder between each pair in [`AtomPairState`].
#[derive(Clone, Debug)]
pub struct AtomPairRender {
    color: (f32, f32, f32),
    radius: f32,
}

impl AtomPairRender {
    pub fn new() -> Self {
        Self {
            color: (1.0, 0.0, 0.0),
            radius: vdw_radius("C") * 0.2,
        }
    }

    pub fn set_color(&mut self, color: (f32, f32, f32)) {
        self.color = color;
    }

    /// Cylinder radius in nanometers.
    pub fn set_radius(&mut self, radius: f32) {
        self.radius = radius;
    }
}

impl Default for AtomPairRender {
    fn default() -> Self {
        Self::new()
    }
}

impl AdditionalRender for AtomPairRender {
    fn update_scene(&self, scene: &mut Scene, frame_state: &RenderFrameState<'_>) {
        let Some(molecule) = frame_state.molecule else {
            return;
        };
        let Some(states) = frame_state.shared_states else {
            return;
        };

        let color = (self.color.0, self.color.1, self.color.2, 1.0);
        with_state_by_type::<AtomPairState, ()>(states, |state| {
            for &(a, b) in &state.pairs {
                let (Some(atom_a), Some(atom_b)) =
                    (molecule.atoms.get(a), molecule.atoms.get(b))
                else {
                    continue;
                };

                // A zero-length cylinder has no axis to orient it along.
                if (atom_b.position - atom_a.position).magnitude() < 1.0e-6 {
                    continue;
                }

                self.add_cylinder(
                    scene,
                    atom_a.position,
                    atom_b.position,
                    self.radius,
                    color,
                );
            }
        });
    }
}
