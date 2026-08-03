//! Point clouds drawn as small 3-axis crosses.

use crate::frame_state::RenderFrameState;
use crate::render_state::with_state_by_type;
use crate::scene_types::{Entity, Mesh, Scene, Vertex};
use crate::{AdditionalRender, GpuPipeline};
use lin_alg::f32::{Quaternion, Vec3};

/// One cloud's points (in nanometers) and the colour they are drawn in.
#[derive(Clone, Debug, Default)]
pub struct PointCloudLayer {
    pub positions: Vec<Vec3>,
    pub color: (f32, f32, f32),
}

/// The point clouds currently drawn. Several can be overlaid at once, each with
/// its own colour, so multiple sources can be told apart.
#[derive(Clone, Debug, Default)]
pub struct PointCloudState {
    pub layers: Vec<PointCloudLayer>,
}

/// Draws each point as a small 3-axis cross on the wireframe pipeline, so a
/// cloud reads as see-through structure regardless of the viewer's render style.
///
/// Suited to dot surfaces (the Connolly / solvent-accessible point sets that
/// tools like `gmx sasa` emit), grid points, or any sampling the application
/// wants to show without occluding the molecule.
pub struct PointCloudRender {
    /// Half-length of each cross arm, in nanometers.
    radius: f32,
}

impl PointCloudRender {
    pub fn new() -> Self {
        Self { radius: 0.04 }
    }

    pub fn set_radius(&mut self, radius: f32) {
        self.radius = radius;
    }

    /// Unit cross mesh: three axis-aligned segments through the origin.
    ///
    /// The wireframe pipeline draws its vertex stream as a `LineList` (vertex
    /// pairs), and [`Scene`] geometry reaches it as triangles emitted three
    /// vertices at a time. Laying the six endpoints out in this index order
    /// makes the emitted stream pair up as `(-x,+x)`, `(-y,+y)`, `(-z,+z)` — the
    /// three cross arms — with no stray connecting lines.
    fn unit_cross_mesh() -> Mesh {
        let n = Vec3::new(0.0, 1.0, 0.0);
        Mesh {
            vertices: vec![
                Vertex::new([-1.0, 0.0, 0.0], n),
                Vertex::new([1.0, 0.0, 0.0], n),
                Vertex::new([0.0, -1.0, 0.0], n),
                Vertex::new([0.0, 1.0, 0.0], n),
                Vertex::new([0.0, 0.0, -1.0], n),
                Vertex::new([0.0, 0.0, 1.0], n),
            ],
            indices: vec![0, 1, 2, 3, 4, 5],
        }
    }
}

impl Default for PointCloudRender {
    fn default() -> Self {
        Self::new()
    }
}

impl AdditionalRender for PointCloudRender {
    /// One cross per point, and a solvent-accessible surface is hundreds of
    /// thousands of them -- so re-deriving this on every frame the camera
    /// merely moved is real CPU time. Reporting a revision lets the renderer
    /// keep what it has.
    fn revision(&self, frame: &RenderFrameState<'_>) -> Option<u64> {
        frame.overlay_revision::<PointCloudState>()
    }

    fn gpu_pipeline(&self) -> GpuPipeline {
        GpuPipeline::Wireframe
    }

    fn update_scene(&self, scene: &mut Scene, frame_state: &RenderFrameState<'_>) {
        let Some(states) = frame_state.shared_states else {
            return;
        };

        with_state_by_type::<PointCloudState, ()>(states, |state| {
            let total: usize = state.layers.iter().map(|l| l.positions.len()).sum();
            if total == 0 {
                return;
            }

            // All crosses share one mesh; per-point placement, scale and colour
            // live on the entities.
            let mesh_idx = scene.meshes.len();
            scene.meshes.push(Self::unit_cross_mesh());

            scene.entities.reserve(total);
            for layer in &state.layers {
                for &position in &layer.positions {
                    scene.entities.push(Entity::new(
                        mesh_idx,
                        position,
                        Quaternion::new_identity(),
                        self.radius,
                        (layer.color.0, layer.color.1, layer.color.2, 1.0),
                        0.1,
                    ));
                }
            }
        });
    }
}
