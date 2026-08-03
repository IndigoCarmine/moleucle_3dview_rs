//! Periodic simulation box drawn as wireframe edges.

use crate::frame_state::RenderFrameState;
use crate::render_state::with_state_by_type;
use crate::scene_types::Scene;
use crate::{AdditionalRender, GpuPipeline};
use lin_alg::f32::Vec3;

/// The simulation box, in nanometers.
///
/// The box is a parallelepiped spanned by three cell vectors from `origin`.
/// That covers the rectangular case and the triclinic ones a rhombic
/// dodecahedron or truncated octahedron is stored as — which is what GROMACS
/// writes for most solvated systems, so it is not an exotic case.
///
/// This is also what [`crate::PeriodicImages`] replicates along, so a cell that
/// describes the real box makes the periodic images land in the right places.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimulationCellState {
    /// Corner the box extends from. GROMACS puts this at the origin.
    pub origin: Vec3,
    /// The three cell vectors, i.e. the columns of the box matrix. All-zero
    /// means "no box"; nothing is drawn.
    pub vectors: [Vec3; 3],
}

impl SimulationCellState {
    /// An axis-aligned box with the given edge lengths.
    pub fn rectangular(size: Vec3) -> Self {
        Self::triclinic([
            Vec3::new(size.x, 0.0, 0.0),
            Vec3::new(0.0, size.y, 0.0),
            Vec3::new(0.0, 0.0, size.z),
        ])
    }

    /// A box spanned by three arbitrary cell vectors.
    pub fn triclinic(vectors: [Vec3; 3]) -> Self {
        Self {
            origin: Vec3::new(0.0, 0.0, 0.0),
            vectors,
        }
    }

    pub fn with_origin(mut self, origin: Vec3) -> Self {
        self.origin = origin;
        self
    }

    /// Whether there is a box to draw at all.
    pub fn is_empty(&self) -> bool {
        self.vectors
            .iter()
            .all(|v| v.magnitude_squared() <= f32::EPSILON)
    }

    /// The corner reached by taking each cell vector or not.
    pub fn corner(&self, steps: [bool; 3]) -> Vec3 {
        let mut corner = self.origin;
        for (vector, take) in self.vectors.iter().zip(steps) {
            if take {
                corner += *vector;
            }
        }
        corner
    }

    /// Translation carrying the primary cell onto periodic image `(i, j, k)`.
    pub fn image_translation(&self, image: [i32; 3]) -> Vec3 {
        let mut translation = Vec3::new(0.0, 0.0, 0.0);
        for (vector, count) in self.vectors.iter().zip(image) {
            translation += *vector * count as f32;
        }
        translation
    }

    /// The box's twelve edges as endpoint pairs.
    ///
    /// Each edge runs along one cell vector between two corners that agree on
    /// the other two, which is the definition of a parallelepiped's edges and
    /// works for triclinic cells without special-casing.
    pub fn edges(&self) -> [[Vec3; 2]; 12] {
        let mut edges = [[Vec3::new(0.0, 0.0, 0.0); 2]; 12];
        let mut next = 0;

        for axis in 0..3 {
            let (other_a, other_b) = match axis {
                0 => (1, 2),
                1 => (0, 2),
                _ => (0, 1),
            };
            for a in [false, true] {
                for b in [false, true] {
                    let mut from = [false; 3];
                    from[other_a] = a;
                    from[other_b] = b;
                    let mut to = from;
                    to[axis] = true;

                    edges[next] = [self.corner(from), self.corner(to)];
                    next += 1;
                }
            }
        }

        edges
    }
}

impl Default for SimulationCellState {
    fn default() -> Self {
        Self::rectangular(Vec3::new(0.0, 0.0, 0.0))
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

        // `SimulationCellState` is four `Vec3`s, so snapshot it and let the lock
        // go before building geometry.
        let Some(state) = with_state_by_type::<SimulationCellState, _>(states, |state| *state)
        else {
            return;
        };
        if state.is_empty() {
            return;
        }

        let color = (self.color.0, self.color.1, self.color.2, 1.0);
        for [start, end] in state.edges() {
            self.add_cylinder(scene, start, end, self.edge_radius, color);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_rectangular_box_has_twelve_axis_aligned_edges() {
        let cell = SimulationCellState::rectangular(Vec3::new(2.0, 3.0, 5.0));
        let edges = cell.edges();
        assert_eq!(edges.len(), 12);

        // Four edges along each axis, each the length of that edge.
        for (axis, expected) in [(0usize, 2.0_f32), (1, 3.0), (2, 5.0)] {
            let along = edges
                .iter()
                .filter(|[from, to]| {
                    let d = *to - *from;
                    let components = [d.x, d.y, d.z];
                    (components[axis] - expected).abs() < 1e-5
                        && components
                            .iter()
                            .enumerate()
                            .all(|(i, c)| i == axis || c.abs() < 1e-5)
                })
                .count();
            assert_eq!(along, 4, "expected four edges along axis {axis}");
        }
    }

    #[test]
    fn a_triclinic_box_keeps_its_skew() {
        // A cell whose second vector leans into x, as a rhombic dodecahedron's
        // does. The edges along it must lean the same way.
        let cell = SimulationCellState::triclinic([
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(0.0, 0.0, 2.0),
        ]);

        let leaning = cell
            .edges()
            .iter()
            .filter(|[from, to]| {
                let d = *to - *from;
                (d.x - 1.0).abs() < 1e-5 && (d.y - 2.0).abs() < 1e-5 && d.z.abs() < 1e-5
            })
            .count();
        assert_eq!(leaning, 4);
    }

    #[test]
    fn image_translations_are_integer_combinations_of_the_cell_vectors() {
        let cell = SimulationCellState::triclinic([
            Vec3::new(2.0, 0.0, 0.0),
            Vec3::new(1.0, 2.0, 0.0),
            Vec3::new(0.0, 0.5, 3.0),
        ]);

        let origin = cell.image_translation([0, 0, 0]);
        assert!(origin.magnitude() < 1e-6, "the primary cell does not move");

        let t = cell.image_translation([1, -1, 2]);
        let expected = Vec3::new(2.0 - 1.0, -2.0 + 1.0, 6.0);
        assert!((t - expected).magnitude() < 1e-5, "got {t:?}");
    }

    #[test]
    fn an_unset_cell_is_empty() {
        assert!(SimulationCellState::default().is_empty());
        assert!(!SimulationCellState::rectangular(Vec3::new(1.0, 1.0, 1.0)).is_empty());
    }
}
