//! Periodic simulation box drawn as wireframe edges.

use crate::frame_state::RenderFrameState;
use crate::render_state::with_state_by_type;
use crate::scene_types::Scene;
use crate::{AdditionalRender, GpuPipeline};
use lin_alg::f32::Vec3;

/// The simulation box to draw, in nanometers.
///
/// A rectangular periodic cell is universal to molecular dynamics, so this lives
/// here rather than in each application. Triclinic cells are not represented;
/// `size` is the box's edge lengths along x, y and z.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimulationCellState {
    /// Corner the box extends from. GROMACS puts this at the origin.
    pub origin: Vec3,
    /// Edge lengths `(x, y, z)`. All-zero means "no box"; nothing is drawn.
    pub size: Vec3,
}

impl SimulationCellState {
    pub fn new(size: Vec3) -> Self {
        Self {
            origin: Vec3::new(0.0, 0.0, 0.0),
            size,
        }
    }

    pub fn with_origin(mut self, origin: Vec3) -> Self {
        self.origin = origin;
        self
    }

    /// Whether there is a box to draw at all.
    pub fn is_empty(&self) -> bool {
        self.size.x <= 0.0 && self.size.y <= 0.0 && self.size.z <= 0.0
    }
}

impl Default for SimulationCellState {
    fn default() -> Self {
        Self::new(Vec3::new(0.0, 0.0, 0.0))
    }
}

/// Draws the twelve edges of [`SimulationCellState`] as thin cylinders on the
/// wireframe pipeline, so the box reads as an outline at any zoom without
/// occluding the structure inside it.
pub struct SimulationCellRender {
    color: (f32, f32, f32),
    edge_radius: f32,
}

impl SimulationCellRender {
    pub fn new() -> Self {
        Self {
            color: (0.5, 0.5, 0.5),
            edge_radius: 0.01,
        }
    }

    pub fn set_color(&mut self, color: (f32, f32, f32)) {
        self.color = color;
    }

    /// Half-thickness of each edge, in nanometers.
    pub fn set_edge_radius(&mut self, radius: f32) {
        self.edge_radius = radius;
    }
}

impl Default for SimulationCellRender {
    fn default() -> Self {
        Self::new()
    }
}

impl AdditionalRender for SimulationCellRender {
    fn gpu_pipeline(&self) -> GpuPipeline {
        GpuPipeline::Wireframe
    }

    fn update_scene(&self, scene: &mut Scene, frame_state: &RenderFrameState<'_>) {
        let Some(states) = frame_state.shared_states else {
            return;
        };

        // `SimulationCellState` is two `Vec3`s, so snapshot it and let the lock
        // go before building geometry.
        let Some(state) = with_state_by_type::<SimulationCellState, _>(states, |state| *state)
        else {
            return;
        };
        if state.is_empty() {
            return;
        }

        let (o, s) = (state.origin, state.size);
        let corner = |x: bool, y: bool, z: bool| {
            Vec3::new(
                o.x + if x { s.x } else { 0.0 },
                o.y + if y { s.y } else { 0.0 },
                o.z + if z { s.z } else { 0.0 },
            )
        };

        // The twelve edges, grouped by the axis they run along.
        let edges = [
            // along x
            [corner(false, false, false), corner(true, false, false)],
            [corner(false, true, false), corner(true, true, false)],
            [corner(false, false, true), corner(true, false, true)],
            [corner(false, true, true), corner(true, true, true)],
            // along y
            [corner(false, false, false), corner(false, true, false)],
            [corner(true, false, false), corner(true, true, false)],
            [corner(false, false, true), corner(false, true, true)],
            [corner(true, false, true), corner(true, true, true)],
            // along z
            [corner(false, false, false), corner(false, false, true)],
            [corner(true, false, false), corner(true, false, true)],
            [corner(false, true, false), corner(false, true, true)],
            [corner(true, true, false), corner(true, true, true)],
        ];

        let color = (self.color.0, self.color.1, self.color.2, 1.0);
        for [start, end] in edges {
            self.add_cylinder(scene, start, end, self.edge_radius, color);
        }
    }
}
