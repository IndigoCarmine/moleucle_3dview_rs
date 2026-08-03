//! Colour highlight for named groups of atoms.

use crate::frame_state::RenderFrameState;
use crate::render_state::with_state_by_type;
use crate::scene_types::Scene;
use crate::{vdw_radius, AdditionalRender, GpuPipeline};

/// One group's atoms, by index into the rendered molecule, and its colour.
#[derive(Clone, Debug)]
pub struct AtomGroup {
    pub atom_indices: Vec<usize>,
    pub color: (f32, f32, f32),
}

/// The atom groups currently highlighted.
///
/// Overlapping groups stack spheres at the same position, so callers that want
/// each atom to belong to at most one group should resolve that before setting
/// the state.
#[derive(Clone, Debug)]
pub struct AtomGroupState {
    pub groups: Vec<AtomGroup>,
    pub visible: bool,
    /// Alpha applied to every group's spheres, in `0.0..=1.0`. Independent of
    /// the main molecule's opacity, so a highlight can stay solid over a faded
    /// structure or the other way round.
    pub opacity: f32,
}

impl Default for AtomGroupState {
    fn default() -> Self {
        Self {
            groups: Vec::new(),
            visible: false,
            opacity: 1.0,
        }
    }
}

/// Draws each group's atoms as coloured sphere impostors, which stays cheap for
/// groups covering a large fraction of a big system.
#[derive(Clone, Debug)]
pub struct AtomGroupRender {
    radius: f32,
}

impl AtomGroupRender {
    pub fn new() -> Self {
        Self {
            radius: vdw_radius("C") * 0.5,
        }
    }

    /// Sphere radius in nanometers.
    pub fn set_radius(&mut self, radius: f32) {
        self.radius = radius;
    }
}

impl Default for AtomGroupRender {
    fn default() -> Self {
        Self::new()
    }
}

impl AdditionalRender for AtomGroupRender {
    /// One sphere per atom in every group, and a group can cover the whole
    /// system -- so re-deriving this on every frame the camera merely moved is
    /// real CPU time. Reporting a revision lets the renderer keep what it has.
    fn revision(&self, frame: &RenderFrameState<'_>) -> Option<u64> {
        frame.overlay_revision::<AtomGroupState>()
    }

    fn gpu_pipeline(&self) -> GpuPipeline {
        GpuPipeline::SphereImpostor
    }

    fn update_scene(&self, scene: &mut Scene, frame_state: &RenderFrameState<'_>) {
        let Some(molecule) = frame_state.molecule else {
            return;
        };
        let Some(states) = frame_state.shared_states else {
            return;
        };

        with_state_by_type::<AtomGroupState, ()>(states, |state| {
            if !state.visible {
                return;
            }
            let alpha = state.opacity.clamp(0.0, 1.0);
            if alpha <= 0.0 {
                return;
            }

            for group in &state.groups {
                for &atom_index in &group.atom_indices {
                    let Some(atom) = molecule.atoms.get(atom_index) else {
                        continue;
                    };

                    self.add_sphere(
                        scene,
                        frame_state,
                        atom.position,
                        self.radius,
                        (group.color.0, group.color.1, group.color.2, alpha),
                    );
                }
            }
        });
    }
}
