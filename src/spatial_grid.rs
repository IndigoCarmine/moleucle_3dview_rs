//! Uniform spatial grid over atom positions, in compressed-row (CSR) layout.
//!
//! Two things in this crate need "which atoms are near here?" over hundreds of
//! thousands to millions of atoms: bond inference at load time, and ray picking
//! on every pointer move. Both were linear scans over every atom.
//!
//! The grid is stored as a prefix-sum `cell_starts` plus a flat `atom_ids`
//! array rather than a `HashMap<(i32, i32, i32), Vec<usize>>`: one allocation
//! each instead of one `Vec` per occupied cell, contiguous iteration, and no
//! hashing per lookup. At GROMACS scale that difference is most of the point.

use crate::molecule::Atom;
use lin_alg::f32::Vec3;

/// Upper bound on total cells, so a structure with one far-flung atom cannot
/// turn a reasonable cell size into a gigabyte of empty buckets. The cell size
/// is grown until the grid fits.
const MAX_CELLS: usize = 1 << 22;

/// The part of the grid that maps a position to a cell. Split out so the
/// counting-sort passes in `build` can use it while mutating the arrays.
#[derive(Clone, Copy)]
struct Layout {
    origin: Vec3,
    cell_size: f32,
    inv_cell: f32,
    dims: [usize; 3],
}

impl Layout {
    fn cell_count(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Grid coordinates of `p`, clamped into range.
    fn coords_of(&self, p: Vec3) -> [usize; 3] {
        let axis = |v: f32, origin: f32, dim: usize| -> usize {
            let raw = ((v - origin) * self.inv_cell).floor();
            if raw < 0.0 {
                0
            } else {
                (raw as usize).min(dim - 1)
            }
        };
        [
            axis(p.x, self.origin.x, self.dims[0]),
            axis(p.y, self.origin.y, self.dims[1]),
            axis(p.z, self.origin.z, self.dims[2]),
        ]
    }

    #[inline]
    fn cell_index(&self, c: [usize; 3]) -> usize {
        (c[2] * self.dims[1] + c[1]) * self.dims[0] + c[0]
    }

    /// Visit every cell the axis-aligned box of `radius` around `position`
    /// touches. With `radius == 0.0` that is the single cell containing it.
    fn for_each_overlapped_cell(&self, position: Vec3, radius: f32, mut f: impl FnMut(usize)) {
        let lo = self.coords_of(Vec3::new(
            position.x - radius,
            position.y - radius,
            position.z - radius,
        ));
        let hi = self.coords_of(Vec3::new(
            position.x + radius,
            position.y + radius,
            position.z + radius,
        ));

        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    f(self.cell_index([x, y, z]));
                }
            }
        }
    }
}

pub(crate) struct SpatialGrid {
    layout: Layout,
    /// `cell_starts[c]..cell_starts[c + 1]` indexes `atom_ids` for cell `c`.
    cell_starts: Vec<u32>,
    atom_ids: Vec<u32>,
}

impl SpatialGrid {
    /// Grid holding each atom in exactly the cell containing its centre.
    ///
    /// Pair with [`Self::for_each_near`], which searches the 3x3x3 block of
    /// cells and is therefore exact as long as `cell_size` is at least the
    /// largest interaction distance.
    pub(crate) fn points(atoms: &[Atom], cell_size: f32) -> Self {
        Self::build(atoms, cell_size, |_| 0.0)
    }

    /// Grid holding each atom in *every* cell its bounding sphere overlaps.
    ///
    /// Pair with [`Self::for_each_along_ray`]: because a sphere is listed in
    /// every cell it reaches into, a ray only has to look at the cells it
    /// actually passes through — no neighbourhood dilation, and no chance of
    /// missing a sphere whose centre sits in a cell the ray misses.
    pub(crate) fn spheres(
        atoms: &[Atom],
        cell_size: f32,
        radius_of: impl Fn(usize) -> f32,
    ) -> Self {
        Self::build(atoms, cell_size, radius_of)
    }

    fn build(atoms: &[Atom], cell_size: f32, radius_of: impl Fn(usize) -> f32) -> Self {
        let radii: Vec<f32> = (0..atoms.len()).map(&radius_of).collect();
        let layout = Self::layout_for(atoms, &radii, cell_size);

        let mut cell_starts = vec![0u32; layout.cell_count() + 1];
        let mut entries = 0usize;
        for (index, atom) in atoms.iter().enumerate() {
            layout.for_each_overlapped_cell(atom.position, radii[index], |cell| {
                cell_starts[cell] += 1;
                entries += 1;
            });
        }

        // Prefix-sum the counts into start offsets.
        let mut running = 0u32;
        for slot in cell_starts.iter_mut() {
            let count = *slot;
            *slot = running;
            running += count;
        }

        let mut atom_ids = vec![0u32; entries];
        let mut cursor = cell_starts.clone();
        for (index, atom) in atoms.iter().enumerate() {
            layout.for_each_overlapped_cell(atom.position, radii[index], |cell| {
                atom_ids[cursor[cell] as usize] = index as u32;
                cursor[cell] += 1;
            });
        }

        Self {
            layout,
            cell_starts,
            atom_ids,
        }
    }

    fn layout_for(atoms: &[Atom], radii: &[f32], cell_size: f32) -> Layout {
        let mut min = Vec3::new(0.0, 0.0, 0.0);
        let mut max = min;
        if let Some(first) = atoms.first() {
            min = first.position;
            max = first.position;
            for atom in &atoms[1..] {
                let p = atom.position;
                min.x = min.x.min(p.x);
                min.y = min.y.min(p.y);
                min.z = min.z.min(p.z);
                max.x = max.x.max(p.x);
                max.y = max.y.max(p.y);
                max.z = max.z.max(p.z);
            }
        }

        // Pad by the largest radius so no sphere reaches outside the grid.
        let pad = radii.iter().fold(0.0_f32, |m, &r| m.max(r));
        let origin = Vec3::new(min.x - pad, min.y - pad, min.z - pad);
        let extent = Vec3::new(
            (max.x - min.x) + 2.0 * pad,
            (max.y - min.y) + 2.0 * pad,
            (max.z - min.z) + 2.0 * pad,
        );

        let mut cell_size = cell_size.max(1e-4);
        let mut dims = Self::dims_for(extent, cell_size);
        // Grow the cells rather than the allocation when the structure is
        // sparse relative to its bounding box.
        while dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]) > MAX_CELLS {
            cell_size *= 2.0;
            dims = Self::dims_for(extent, cell_size);
        }

        Layout {
            origin,
            cell_size,
            inv_cell: 1.0 / cell_size,
            dims,
        }
    }

    fn dims_for(extent: Vec3, cell_size: f32) -> [usize; 3] {
        let axis = |len: f32| {
            let cells = (len / cell_size).ceil();
            if cells.is_finite() && cells >= 1.0 {
                cells as usize
            } else {
                1
            }
        };
        [axis(extent.x), axis(extent.y), axis(extent.z)]
    }

    /// The grid's axis-aligned extent, which contains every atom it holds
    /// (padded by the radii it was built with).
    pub(crate) fn bounds(&self) -> (Vec3, Vec3) {
        let layout = self.layout;
        let size = layout.cell_size;
        (
            layout.origin,
            Vec3::new(
                layout.origin.x + layout.dims[0] as f32 * size,
                layout.origin.y + layout.dims[1] as f32 * size,
                layout.origin.z + layout.dims[2] as f32 * size,
            ),
        )
    }

    #[inline]
    fn atoms_in(&self, cell: usize) -> &[u32] {
        let start = self.cell_starts[cell] as usize;
        let end = self.cell_starts[cell + 1] as usize;
        &self.atom_ids[start..end]
    }

    /// Call `f` with every atom in the 3x3x3 block of cells around `position`.
    ///
    /// Atoms outside the interaction distance are still reported — this narrows
    /// the candidate set, it does not answer the query.
    pub(crate) fn for_each_near(&self, position: Vec3, mut f: impl FnMut(u32)) {
        let dims = self.layout.dims;
        let c = self.layout.coords_of(position);
        let lo = [
            c[0].saturating_sub(1),
            c[1].saturating_sub(1),
            c[2].saturating_sub(1),
        ];
        let hi = [
            (c[0] + 1).min(dims[0] - 1),
            (c[1] + 1).min(dims[1] - 1),
            (c[2] + 1).min(dims[2] - 1),
        ];

        for z in lo[2]..=hi[2] {
            for y in lo[1]..=hi[1] {
                for x in lo[0]..=hi[0] {
                    for &id in self.atoms_in(self.layout.cell_index([x, y, z])) {
                        f(id);
                    }
                }
            }
        }
    }

    /// Walk the cells a ray passes through, nearest first (Amanatides & Woo).
    ///
    /// `f` receives each cell's atoms along with the ray parameter at which the
    /// ray *enters* that cell, and returns `false` to stop. That parameter is
    /// what makes early exit sound: once a hit closer than the current cell's
    /// entry distance has been found, no later cell can improve on it.
    ///
    /// `direction` must be normalised, so the parameter is a distance.
    pub(crate) fn for_each_along_ray(
        &self,
        origin: Vec3,
        direction: Vec3,
        mut f: impl FnMut(&[u32], f32) -> bool,
    ) {
        let layout = self.layout;
        let grid_max = Vec3::new(
            layout.origin.x + layout.dims[0] as f32 * layout.cell_size,
            layout.origin.y + layout.dims[1] as f32 * layout.cell_size,
            layout.origin.z + layout.dims[2] as f32 * layout.cell_size,
        );

        // Slab test against the grid's bounding box.
        let (mut t_enter, mut t_exit) = (0.0_f32, f32::MAX);
        for axis in 0..3 {
            let (o, d) = (component(origin, axis), component(direction, axis));
            let (lo, hi) = (component(layout.origin, axis), component(grid_max, axis));
            if d.abs() < 1e-9 {
                if o < lo || o > hi {
                    return;
                }
                continue;
            }
            let (mut t0, mut t1) = ((lo - o) / d, (hi - o) / d);
            if t0 > t1 {
                std::mem::swap(&mut t0, &mut t1);
            }
            t_enter = t_enter.max(t0);
            t_exit = t_exit.min(t1);
            if t_enter > t_exit {
                return;
            }
        }

        let entry = origin + direction * t_enter;
        let start = layout.coords_of(entry);
        let mut cell = [start[0] as isize, start[1] as isize, start[2] as isize];
        let mut step = [0isize; 3];
        let mut t_max = [f32::INFINITY; 3];
        let mut t_delta = [f32::INFINITY; 3];

        for axis in 0..3 {
            let d = component(direction, axis);
            if d.abs() < 1e-9 {
                continue;
            }
            let cell_lo = component(layout.origin, axis) + cell[axis] as f32 * layout.cell_size;
            let boundary = if d > 0.0 {
                cell_lo + layout.cell_size
            } else {
                cell_lo
            };
            step[axis] = if d > 0.0 { 1 } else { -1 };
            t_max[axis] = (boundary - component(origin, axis)) / d;
            t_delta[axis] = layout.cell_size / d.abs();
        }

        let mut t_cell = t_enter;
        loop {
            if (0..3).any(|a| cell[a] < 0 || cell[a] as usize >= layout.dims[a]) {
                return;
            }

            let index =
                layout.cell_index([cell[0] as usize, cell[1] as usize, cell[2] as usize]);
            if !f(self.atoms_in(index), t_cell) {
                return;
            }

            // Advance across whichever boundary comes first.
            let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
                0
            } else if t_max[1] <= t_max[2] {
                1
            } else {
                2
            };
            if step[axis] == 0 || !t_max[axis].is_finite() || t_max[axis] > t_exit {
                return;
            }
            t_cell = t_max[axis];
            cell[axis] += step[axis];
            t_max[axis] += t_delta[axis];
        }
    }
}

#[inline]
fn component(v: Vec3, axis: usize) -> f32 {
    match axis {
        0 => v.x,
        1 => v.y,
        _ => v.z,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::Element;

    /// Deterministic pseudo-random point cloud. `Math.random` equivalents are
    /// avoided so a failure is reproducible.
    fn scattered_atoms(count: usize) -> Vec<Atom> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f32 / (1u64 << 53) as f32
        };

        (0..count)
            .map(|id| Atom {
                position: Vec3::new(
                    next() * 8.0 - 4.0,
                    next() * 8.0 - 4.0,
                    next() * 8.0 - 4.0,
                ),
                element: Element::new("C"),
                id,
                meta: None,
            })
            .collect()
    }

    fn ray_sphere_t(origin: Vec3, dir: Vec3, center: Vec3, radius: f32) -> Option<f32> {
        let oc = origin - center;
        let b = 2.0 * oc.dot(dir);
        let c = oc.magnitude_squared() - radius * radius;
        let disc = b * b - 4.0 * c;
        if disc < 0.0 {
            return None;
        }
        let t = (-b - disc.sqrt()) * 0.5;
        (t > 0.0).then_some(t)
    }

    /// Closest hit by exhaustive search — the answer the grid has to reproduce.
    fn brute_force(atoms: &[Atom], radius: f32, origin: Vec3, dir: Vec3) -> Option<usize> {
        let mut best: Option<(f32, usize)> = None;
        for (i, atom) in atoms.iter().enumerate() {
            if let Some(t) = ray_sphere_t(origin, dir, atom.position, radius) {
                if best.is_none_or(|(bt, _)| t < bt) {
                    best = Some((t, i));
                }
            }
        }
        best.map(|(_, i)| i)
    }

    /// Same, through the grid: walk cells nearest-first and stop once no
    /// unvisited cell can hold anything closer.
    fn grid_pick(grid: &SpatialGrid, atoms: &[Atom], radius: f32, origin: Vec3, dir: Vec3)
        -> Option<usize>
    {
        let mut best: Option<(f32, usize)> = None;
        grid.for_each_along_ray(origin, dir, |candidates, cell_entry_t| {
            if best.is_some_and(|(bt, _)| bt < cell_entry_t) {
                return false;
            }
            for &candidate in candidates {
                let i = candidate as usize;
                if let Some(t) = ray_sphere_t(origin, dir, atoms[i].position, radius) {
                    if best.is_none_or(|(bt, _)| t < bt) {
                        best = Some((t, i));
                    }
                }
            }
            true
        });
        best.map(|(_, i)| i)
    }

    #[test]
    fn ray_traversal_agrees_with_brute_force() {
        let atoms = scattered_atoms(600);
        let radius = 0.15_f32;
        let grid = SpatialGrid::spheres(&atoms, radius * 4.0, |_| radius);

        // Rays from every octant, plus three exactly axis-aligned ones (the
        // degenerate case for the DDA's per-axis setup).
        let mut directions = Vec::new();
        for x in [-1.0_f32, 0.0, 1.0] {
            for y in [-1.0_f32, 0.0, 1.0] {
                for z in [-1.0_f32, 0.0, 1.0] {
                    if x == 0.0 && y == 0.0 && z == 0.0 {
                        continue;
                    }
                    directions.push(Vec3::new(x, y, z).to_normalized());
                }
            }
        }

        let mut hits = 0;
        for (index, dir) in directions.iter().enumerate() {
            // Start well outside the cloud and aim back through it.
            let offset = 0.37 * index as f32 - 2.0;
            let origin = *dir * -12.0 + Vec3::new(offset * 0.1, offset * 0.07, offset * 0.05);

            let expected = brute_force(&atoms, radius, origin, *dir);
            let actual = grid_pick(&grid, &atoms, radius, origin, *dir);
            assert_eq!(expected, actual, "ray {index} from {origin:?} along {dir:?}");
            if expected.is_some() {
                hits += 1;
            }
        }

        assert!(hits > 0, "the test rays should actually hit something");
    }

    #[test]
    fn a_ray_that_misses_the_grid_visits_nothing() {
        let atoms = scattered_atoms(50);
        let grid = SpatialGrid::spheres(&atoms, 0.5, |_| 0.1);

        let mut visited = 0;
        grid.for_each_along_ray(
            Vec3::new(1000.0, 1000.0, 1000.0),
            Vec3::new(0.0, 0.0, 1.0),
            |_, _| {
                visited += 1;
                true
            },
        );
        assert_eq!(visited, 0);
    }

    #[test]
    fn an_empty_grid_is_usable() {
        let grid = SpatialGrid::spheres(&[], 0.5, |_| 0.1);
        grid.for_each_near(Vec3::new(0.0, 0.0, 0.0), |_| panic!("no atoms to report"));
        grid.for_each_along_ray(
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            |atoms, _| {
                assert!(atoms.is_empty());
                true
            },
        );
    }

    /// `for_each_near` must report every atom within one cell, which is what
    /// makes the 3x3x3 search exact for bond inference.
    #[test]
    fn neighbour_search_covers_everything_within_a_cell() {
        let atoms = scattered_atoms(400);
        let cell = 0.9_f32;
        let grid = SpatialGrid::points(&atoms, cell);

        for (i, atom) in atoms.iter().enumerate() {
            let mut reported = std::collections::HashSet::new();
            grid.for_each_near(atom.position, |id| {
                reported.insert(id as usize);
            });

            for (j, other) in atoms.iter().enumerate() {
                let d = other.position - atom.position;
                if d.magnitude() <= cell {
                    assert!(
                        reported.contains(&j),
                        "atom {j} is within one cell of {i} but was not reported"
                    );
                }
            }
        }
    }
}
