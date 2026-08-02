//! X/Y/Z orientation triad.

use crate::frame_state::RenderFrameState;
use crate::render_state::with_state_by_type;
use crate::scene_types::Scene;
use crate::{AdditionalRender, GpuPipeline};
use lin_alg::f32::Vec3;

/// Whether the triad is drawn, and how long each arm is.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AxesState {
    pub visible: bool,
    /// Arm length per axis, in nanometers. `None` sizes the triad from the
    /// molecule's bounding radius.
    ///
    /// Applications with a simulation box usually want to pass the box's edge
    /// lengths here so each coloured arm runs along the edge leaving the origin.
    /// This is an explicit input rather than something read from
    /// [`super::SimulationCellState`] so the two overlays stay independent —
    /// and so this one never has to take a second lock on the state map.
    pub length: Option<Vec3>,
}

impl Default for AxesState {
    fn default() -> Self {
        Self {
            visible: true,
            length: None,
        }
    }
}

/// Draws an X/Y/Z triad at the world origin: three solid arrows following the
/// universal colour convention (X red, Y green, Z blue), each capped with a
/// small ball marking its positive direction.
///
/// Drawn on the triangle pipeline so it stands out against wireframe geometry.
pub struct AxesRender;

impl AxesRender {
    pub fn new() -> Self {
        Self
    }
}

impl Default for AxesRender {
    fn default() -> Self {
        Self::new()
    }
}

impl AdditionalRender for AxesRender {
    fn gpu_pipeline(&self) -> GpuPipeline {
        GpuPipeline::Triangles
    }

    fn update_scene(&self, scene: &mut Scene, frame_state: &RenderFrameState<'_>) {
        let Some(states) = frame_state.shared_states else {
            return;
        };

        // An absent state counts as visible with no explicit length, so simply
        // registering this overlay shows the axes.
        let state = with_state_by_type::<AxesState, _>(states, |state| *state).unwrap_or_default();
        if !state.visible {
            return;
        }

        let fallback = frame_state
            .molecule
            .map(|molecule| molecule.radius())
            .filter(|radius| *radius > 0.0)
            .unwrap_or(2.0);
        let axis_len = |edge: f32| if edge > 0.0 { edge } else { fallback };
        let requested = state.length.unwrap_or(Vec3::new(0.0, 0.0, 0.0));
        let (lx, ly, lz) = (
            axis_len(requested.x),
            axis_len(requested.y),
            axis_len(requested.z),
        );

        // Scale the shaft/tip to the triad so it reads at any zoom, with a small
        // floor so it never vanishes on a tiny structure.
        let max_len = lx.max(ly).max(lz);
        let shaft_radius = (max_len * 0.012).max(0.02);
        let tip_radius = (max_len * 0.03).max(0.05);
        let origin = Vec3::new(0.0, 0.0, 0.0);

        let axes = [
            (Vec3::new(lx, 0.0, 0.0), (0.90, 0.20, 0.20)), // X — red
            (Vec3::new(0.0, ly, 0.0), (0.20, 0.75, 0.25)), // Y — green
            (Vec3::new(0.0, 0.0, lz), (0.25, 0.45, 0.95)), // Z — blue
        ];
        for (end, (r, g, b)) in axes {
            self.add_cylinder(scene, origin, end, shaft_radius, (r, g, b, 1.0));
            self.add_sphere(scene, frame_state, end, tip_radius, (r, g, b, 1.0));
        }
    }
}
