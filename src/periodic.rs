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
    /// How many cells to draw along each cell vector, the primary cell
    /// included. `[1, 1, 1]` draws only the original; `[3, 3, 3]` draws it
    /// surrounded by its 26 neighbours; `[2, 2, 2]` draws an eight-cell block.
    ///
    /// A total rather than a per-side count, so every size is reachable — a
    /// per-side count can only ever produce odd numbers of cells.
    ///
    /// Per-axis because systems are often periodic in only some directions — a
    /// membrane or a slab wants replication in the plane but not through it.
    pub cells: [u32; 3],
}

impl PeriodicImages {
    /// Draw `cells` cells along each of `cell`'s vectors, the primary cell
    /// included. A count of zero is treated as one.
    pub fn new(cell: SimulationCellState, cells: [u32; 3]) -> Self {
        Self {
            cell,
            cells: cells.map(|n| n.max(1)),
        }
    }

    /// Whether anything beyond the primary cell would be drawn.
    pub fn is_trivial(&self) -> bool {
        self.cell.is_empty() || self.cells.iter().all(|n| *n <= 1)
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
        let cells = if self.cell.is_empty() {
            [1, 1, 1]
        } else {
            self.cells
        };

        std::iter::once([0, 0, 0])
            .chain(
                Self::span(cells[0])
                    .flat_map(move |i| {
                        Self::span(cells[1])
                            .flat_map(move |j| Self::span(cells[2]).map(move |k| [i, j, k]))
                    })
                    .filter(|image| *image != [0, 0, 0]),
            )
            .take(MAX_PERIODIC_IMAGES)
    }

    /// The image indices covering `cells` cells along one axis.
    ///
    /// Kept as centred on the primary cell as the count allows, so an odd count
    /// surrounds it evenly and an even one puts the spare cell on the positive
    /// side. Centring matters because the point of replicating is usually to
    /// see what a structure's *neighbours* look like, and a block growing in one
    /// direction only would leave it in a corner.
    fn span(cells: u32) -> std::ops::RangeInclusive<i32> {
        let n = cells.max(1) as i32;
        -((n - 1) / 2)..=(n / 2)
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
    fn one_cell_per_axis_draws_only_the_primary_cell() {
        let images = PeriodicImages::new(unit_cell(), [1, 1, 1]);
        assert!(images.is_trivial());
        assert_eq!(images.len(), 1);
        assert_eq!(images.indices().collect::<Vec<_>>(), vec![[0, 0, 0]]);

        // A zero count is meaningless; read it as one rather than as "no cells".
        assert_eq!(PeriodicImages::new(unit_cell(), [0, 0, 0]).len(), 1);
    }

    /// The count is a total, so every block size is reachable -- a per-side
    /// count could only ever produce odd numbers of cells.
    #[test]
    fn every_block_size_is_reachable() {
        for n in 1..=6u32 {
            let images = PeriodicImages::new(unit_cell(), [n, 1, 1]);
            assert_eq!(images.len(), n as usize, "{n} cells along a");
        }

        assert_eq!(PeriodicImages::new(unit_cell(), [2, 2, 2]).len(), 8);
        assert_eq!(PeriodicImages::new(unit_cell(), [3, 3, 3]).len(), 27);
        assert_eq!(PeriodicImages::new(unit_cell(), [4, 3, 2]).len(), 24);
    }

    #[test]
    fn counts_are_per_axis() {
        // 3 x 3 x 1: replicate in the plane but not through it.
        let images = PeriodicImages::new(unit_cell(), [3, 3, 1]);
        assert_eq!(images.len(), 9);
        assert!(images.indices().all(|[_, _, k]| k == 0));
    }

    /// An odd block surrounds the primary cell; an even one cannot, and puts
    /// the spare cell on the positive side.
    #[test]
    fn blocks_stay_as_centred_as_the_count_allows() {
        let along_a = |n: u32| {
            let mut indices: Vec<i32> = PeriodicImages::new(unit_cell(), [n, 1, 1])
                .indices()
                .map(|[i, _, _]| i)
                .collect();
            indices.sort_unstable();
            indices
        };

        assert_eq!(along_a(1), vec![0]);
        assert_eq!(along_a(2), vec![0, 1]);
        assert_eq!(along_a(3), vec![-1, 0, 1]);
        assert_eq!(along_a(4), vec![-1, 0, 1, 2]);
        assert_eq!(along_a(5), vec![-2, -1, 0, 1, 2]);
    }

    #[test]
    fn the_primary_cell_comes_first_and_appears_once() {
        let images = PeriodicImages::new(unit_cell(), [3, 3, 3]);
        let indices: Vec<_> = images.indices().collect();

        assert_eq!(indices[0], [0, 0, 0]);
        assert_eq!(indices.iter().filter(|i| **i == [0, 0, 0]).count(), 1);

        let first = images.translations().next().unwrap();
        assert!(first.magnitude() < 1e-6);
    }

    #[test]
    fn translations_follow_the_cell_vectors() {
        let images = PeriodicImages::new(unit_cell(), [3, 1, 1]);
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
        let images = PeriodicImages::new(SimulationCellState::default(), [3, 3, 3]);
        assert!(images.is_trivial());
        assert_eq!(images.len(), 1);
    }
}
