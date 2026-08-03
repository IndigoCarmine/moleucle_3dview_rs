//! Periodic images: drawing the same molecule shifted by whole cell vectors.

use crate::overlays::SimulationCellState;
use lin_alg::f32::Vec3;

/// Upper bound on how many images a single frame will draw.
///
/// Each image is a full redraw of the molecule's geometry, so the cost is in
/// draw calls and fill rate rather than memory — but it is still linear, and a
/// mistyped replication count should degrade rather than lock up the UI. 9x9x9
/// is far past anything worth looking at.
pub const MAX_PERIODIC_IMAGES: usize = 729;

/// How far to replicate the molecule along each cell vector.
///
/// The molecule's geometry is uploaded once and drawn once per image with a
/// different translation, so an image costs no extra vertex memory — only its
/// draw calls. That is the whole reason this lives in the renderer instead of
/// being done by duplicating atoms upstream: at MD scale, 27 copies of a
/// million-atom system is not something a CPU-side replication can afford.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeriodicImages {
    /// The cell being replicated along. Its vectors are what an image is
    /// translated by; see [`SimulationCellState`].
    pub cell: SimulationCellState,
    /// Replicas on *each side* along each cell vector. `[1, 1, 1]` draws the
    /// 3x3x3 block (the original plus its 26 neighbours); `[0, 0, 0]` draws
    /// only the original.
    ///
    /// Per-axis because systems are often periodic in only some directions — a
    /// membrane or a slab wants replication in the plane but not through it.
    pub counts: [u32; 3],
}

impl PeriodicImages {
    /// Replicate `cell` by `counts` on each side of each axis.
    pub fn new(cell: SimulationCellState, counts: [u32; 3]) -> Self {
        Self { cell, counts }
    }

    /// Whether anything beyond the primary cell would be drawn.
    pub fn is_trivial(&self) -> bool {
        self.cell.is_empty() || self.counts == [0, 0, 0]
    }

    /// How many images this describes, including the primary cell, after the
    /// [`MAX_PERIODIC_IMAGES`] clamp.
    pub fn len(&self) -> usize {
        self.translations().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Every image's translation, primary cell first.
    ///
    /// The primary cell leads so that a renderer drawing images in order draws
    /// the real molecule before its copies — which matters for translucency,
    /// where draw order is visible.
    pub fn translations(&self) -> impl Iterator<Item = Vec3> + '_ {
        let cell = self.cell;
        self.indices().map(move |image| cell.image_translation(image))
    }

    /// Every image's `(i, j, k)`, primary cell first, clamped to
    /// [`MAX_PERIODIC_IMAGES`].
    pub fn indices(&self) -> impl Iterator<Item = [i32; 3]> + '_ {
        let counts = if self.cell.is_empty() {
            [0, 0, 0]
        } else {
            self.counts
        };
        let span = |c: u32| -(c as i32)..=(c as i32);

        std::iter::once([0, 0, 0])
            .chain(
                span(counts[0])
                    .flat_map(move |i| {
                        span(counts[1])
                            .flat_map(move |j| span(counts[2]).map(move |k| [i, j, k]))
                    })
                    .filter(|image| *image != [0, 0, 0]),
            )
            .take(MAX_PERIODIC_IMAGES)
    }
}

impl Default for PeriodicImages {
    fn default() -> Self {
        Self::new(SimulationCellState::default(), [0, 0, 0])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cell() -> SimulationCellState {
        SimulationCellState::rectangular(Vec3::new(1.0, 2.0, 4.0))
    }

    #[test]
    fn no_replication_draws_only_the_primary_cell() {
        let images = PeriodicImages::new(unit_cell(), [0, 0, 0]);
        assert!(images.is_trivial());
        assert_eq!(images.len(), 1);
        assert_eq!(images.indices().collect::<Vec<_>>(), vec![[0, 0, 0]]);
    }

    #[test]
    fn counts_are_per_side_and_per_axis() {
        // 3 x 3 x 1
        let images = PeriodicImages::new(unit_cell(), [1, 1, 0]);
        assert_eq!(images.len(), 9);
        assert!(images.indices().all(|[_, _, k]| k == 0));

        // 3 x 3 x 3
        assert_eq!(PeriodicImages::new(unit_cell(), [1, 1, 1]).len(), 27);
        // 5 x 1 x 1
        assert_eq!(PeriodicImages::new(unit_cell(), [2, 0, 0]).len(), 5);
    }

    /// Draw order is visible when the molecule is translucent, so the real
    /// molecule has to come first.
    #[test]
    fn the_primary_cell_comes_first_and_appears_once() {
        let images = PeriodicImages::new(unit_cell(), [1, 1, 1]);
        let indices: Vec<_> = images.indices().collect();

        assert_eq!(indices[0], [0, 0, 0]);
        assert_eq!(indices.iter().filter(|i| **i == [0, 0, 0]).count(), 1);

        let first = images.translations().next().unwrap();
        assert!(first.magnitude() < 1e-6);
    }

    #[test]
    fn translations_follow_the_cell_vectors() {
        let images = PeriodicImages::new(unit_cell(), [1, 0, 0]);
        let mut translations = images.translations();

        assert!(translations.next().unwrap().magnitude() < 1e-6);
        let rest: Vec<f32> = translations.map(|t| t.x).collect();
        assert_eq!(rest.len(), 2);
        assert!(rest.contains(&-1.0) && rest.contains(&1.0));
    }

    /// A mistyped replication count should degrade, not lock up the UI.
    #[test]
    fn the_image_count_is_capped() {
        let images = PeriodicImages::new(unit_cell(), [100, 100, 100]);
        assert_eq!(images.len(), MAX_PERIODIC_IMAGES);
        assert_eq!(images.indices().next(), Some([0, 0, 0]));
    }

    #[test]
    fn a_cell_with_no_box_never_replicates() {
        let images = PeriodicImages::new(SimulationCellState::default(), [2, 2, 2]);
        assert!(images.is_trivial());
        assert_eq!(images.len(), 1);
    }
}
