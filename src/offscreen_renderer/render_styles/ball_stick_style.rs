use lin_alg::f32::{Quaternion, Vec3};

use crate::atom_radii::{ball_stick_radius, default_ball_stick_bond_radius};
use crate::viewer::ColorFn;
use crate::Molecule;

use super::super::{
    append_line, append_mesh_triangles, bond_line_offsets, RenderMesh, Vertex,
    DEFAULT_BOND_CYLINDER_SIDES, MAX_RENDER_VERTICES,
};
use super::{MolecularRenderStyle, StyleBuildContext};

pub(super) const BALL_STICK_STYLE: BallStickStyle = BallStickStyle;

pub(super) struct BallStickStyle;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BallstickQuality {
    High,
    Medium,
    Low,
}

impl MolecularRenderStyle for BallStickStyle {
    fn primitive_stride(&self) -> usize {
        3
    }

    fn build_vertices(
        &self,
        context: &StyleBuildContext<'_>,
        molecule: Option<&Molecule>,
        color_fn: ColorFn,
    ) -> Vec<Vertex> {
        build_ballstick_vertices(context, molecule, color_fn, true)
    }
}

pub(super) fn build_ballstick_vertices(
    context: &StyleBuildContext<'_>,
    molecule: Option<&Molecule>,
    color_fn: ColorFn,
    include_bonds: bool,
) -> Vec<Vertex> {
    let quality = pick_ballstick_quality(context, molecule);
    let mesh_resolution = context.preference.mesh_resolution();
    let quality_resolution = match quality {
        BallstickQuality::High => mesh_resolution.max(3),
        BallstickQuality::Medium => (mesh_resolution / 2).max(3),
        BallstickQuality::Low => 3,
    };
    let low_mode = matches!(quality, BallstickQuality::Low);

    let generated_meshes = if quality_resolution == mesh_resolution {
        None
    } else {
        Some((
            RenderMesh::new_sphere_uv(1.0, quality_resolution, quality_resolution * 2),
            RenderMesh::new_cylinder_open_ended(1.0, 1.0, DEFAULT_BOND_CYLINDER_SIDES),
        ))
    };
    let (sphere_mesh, cylinder_mesh): (&RenderMesh, &RenderMesh) =
        if let Some((sphere, cylinder)) = &generated_meshes {
            (sphere, cylinder)
        } else {
            (context.sphere_mesh, context.cylinder_mesh)
        };

    let max_vertices = MAX_RENDER_VERTICES;
    let mut vertices = if let Some(mol) = molecule {
        let capacity = mol
            .bonds
            .len()
            .saturating_mul(50)
            .saturating_add(mol.atoms.len().saturating_mul(200))
            .saturating_add(225)
            .min(max_vertices);
        Vec::with_capacity(capacity)
    } else {
        Vec::with_capacity(225.min(max_vertices))
    };

    if let Some(mol) = molecule {
        if include_bonds {
            'bonds: for bond in &mol.bonds {
                if !context.is_bond_visible(bond) {
                    continue;
                }
                let Some((a, b)) = mol.bond_endpoints(bond) else {
                    continue;
                };
                let diff = b - a;
                let len = diff.magnitude();
                if len < 0.001 {
                    continue;
                }

                let dir = diff.to_normalized();
                let up = Vec3::new(0.0, 1.0, 0.0);
                let orientation = Quaternion::from_unit_vecs(up, dir);
                let mid = (a + b) * 0.5;

                let bond_order = bond.order.max(1) as usize;
                let line_offsets = bond_line_offsets(bond_order);
                let mut lateral = Vec3::new(1.0, 0.0, 0.0);
                if dir.dot(lateral).abs() > 0.9 {
                    lateral = Vec3::new(0.0, 0.0, 1.0);
                }
                lateral = (lateral - dir * lateral.dot(dir)).to_normalized();

                let base_radius = if bond_order <= 1 {
                    default_ball_stick_bond_radius()
                } else {
                    default_ball_stick_bond_radius() * 0.5
                };
                for offset in line_offsets {
                    if low_mode {
                        if !append_line(
                            &mut vertices,
                            a + lateral * offset,
                            b + lateral * offset,
                            [0.55, 0.55, 0.55, context.molecule_opacity],
                            max_vertices,
                        ) {
                            break 'bonds;
                        }
                    } else if !append_mesh_triangles(
                        &mut vertices,
                        cylinder_mesh,
                        mid + lateral * offset,
                        orientation,
                        Vec3::new(base_radius, len, base_radius),
                        [0.55, 0.55, 0.55, context.molecule_opacity],
                        max_vertices,
                    ) {
                        break 'bonds;
                    }
                }
            }
        }

        'atoms: for (i, atom) in mol.atoms.iter().enumerate() {
            if !context.is_atom_visible(i) {
                continue;
            }
            let pos = atom.position;
            // Per-atom radius / color overrides (e.g. CG beads) when supplied,
            // else the element-derived radius and the color function.
            let radius = context
                .atom_radii
                .and_then(|r| r.get(i).copied())
                .unwrap_or_else(|| ball_stick_radius(&atom.element, false));
            let base = context
                .atom_colors
                .and_then(|c| c.get(i).copied())
                .unwrap_or_else(|| {
                    let c = color_fn(atom, false);
                    [c.0, c.1, c.2, c.3]
                });
            let color = [base[0], base[1], base[2], base[3] * context.molecule_opacity];

            if !append_mesh_triangles(
                &mut vertices,
                sphere_mesh,
                pos,
                Quaternion::new_identity(),
                Vec3::new(radius, radius, radius),
                color,
                max_vertices,
            ) {
                break 'atoms;
            }
        }
    }

    vertices
}

fn pick_ballstick_quality(
    context: &StyleBuildContext<'_>,
    molecule: Option<&Molecule>,
) -> BallstickQuality {
    let Some(mol) = molecule else {
        return BallstickQuality::High;
    };

    let high = estimate_ballstick_vertices(context, mol, BallstickQuality::High);
    if high <= MAX_RENDER_VERTICES {
        return BallstickQuality::High;
    }

    let medium = estimate_ballstick_vertices(context, mol, BallstickQuality::Medium);
    if medium <= MAX_RENDER_VERTICES {
        return BallstickQuality::Medium;
    }

    BallstickQuality::Low
}

fn estimate_ballstick_vertices(
    context: &StyleBuildContext<'_>,
    molecule: &Molecule,
    quality: BallstickQuality,
) -> usize {
    let resolution = match quality {
        BallstickQuality::High => context.preference.mesh_resolution().max(3),
        BallstickQuality::Medium => (context.preference.mesh_resolution() / 2).max(3),
        BallstickQuality::Low => 3,
    };

    let sphere_vertices_per_atom = resolution
        .saturating_mul(resolution.saturating_mul(2))
        .saturating_mul(6);
    let atom_vertices = molecule
        .atoms
        .len()
        .saturating_mul(sphere_vertices_per_atom);

    let bond_vertices = if matches!(quality, BallstickQuality::Low) {
        molecule.bonds.len().saturating_mul(2)
    } else {
        let cylinder_vertices = DEFAULT_BOND_CYLINDER_SIDES.saturating_mul(6);
        let bond_instances = molecule.bonds.iter().fold(0usize, |acc, bond| {
            acc.saturating_add(bond_line_offsets(bond.order.max(1) as usize).len())
        });
        bond_instances.saturating_mul(cylinder_vertices)
    };

    let axis_vertices = if matches!(quality, BallstickQuality::Low) {
        6
    } else {
        (resolution.saturating_mul(2))
            .saturating_mul(12)
            .saturating_mul(3)
    };

    atom_vertices
        .saturating_add(bond_vertices)
        .saturating_add(axis_vertices)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::{Atom, Bond, Element};
    use crate::offscreen_renderer::{OffscreenRendererPreference, DEFAULT_BOND_CYLINDER_SIDES};
    use crate::viewer::default_color_fn;

    /// A short chain: atoms in a line, bonded to their neighbour.
    fn chain(count: usize) -> Molecule {
        let atoms = (0..count)
            .map(|i| Atom {
                position: Vec3::new(i as f32 * 0.15, 0.0, 0.0),
                element: Element::new("C"),
                id: i,
                meta: None,
            })
            .collect();
        let bonds = (0..count.saturating_sub(1))
            .map(|i| Bond {
                atom_a: i,
                atom_b: i + 1,
                order: 1,
            })
            .collect();
        Molecule::from_atoms_bonds(atoms, bonds)
    }

    fn vertex_count(molecule: &Molecule, visible: Option<&[bool]>) -> usize {
        let preference = OffscreenRendererPreference::default();
        let resolution = preference.mesh_resolution();
        let sphere = RenderMesh::new_sphere_uv(1.0, resolution, resolution * 2);
        let cylinder = RenderMesh::new_cylinder_open_ended(1.0, 1.0, DEFAULT_BOND_CYLINDER_SIDES);

        let context = StyleBuildContext {
            preference,
            sphere_mesh: &sphere,
            cylinder_mesh: &cylinder,
            molecule_opacity: 1.0,
            atom_radii: None,
            atom_colors: None,
            visible,
        };

        build_ballstick_vertices(&context, Some(molecule), default_color_fn, true).len()
    }

    /// Per-atom and per-bond vertex costs, derived from the empty/one-atom/
    /// two-atom cases rather than hard-coded, so the test survives a change of
    /// mesh resolution.
    fn costs(molecule_of: impl Fn(usize) -> Molecule) -> (usize, usize) {
        let one = vertex_count(&molecule_of(1), None);
        let two = vertex_count(&molecule_of(2), None);
        // Going from one atom to two adds one atom and one bond.
        (one, two - one - one)
    }

    #[test]
    fn hiding_an_atom_removes_it_and_every_bond_touching_it() {
        let molecule = chain(4);
        let (per_atom, per_bond) = costs(chain);
        assert!(per_atom > 0 && per_bond > 0);

        let full = vertex_count(&molecule, None);
        assert_eq!(full, 4 * per_atom + 3 * per_bond);

        // Hide an interior atom: it takes both of its bonds with it.
        let mut mask = vec![true; 4];
        mask[1] = false;
        assert_eq!(
            vertex_count(&molecule, Some(&mask)),
            3 * per_atom + 1 * per_bond,
        );

        // Hide an end atom: only its single bond goes.
        let mut mask = vec![true; 4];
        mask[3] = false;
        assert_eq!(
            vertex_count(&molecule, Some(&mask)),
            3 * per_atom + 2 * per_bond,
        );
    }

    #[test]
    fn an_all_visible_mask_matches_no_mask() {
        let molecule = chain(4);
        assert_eq!(
            vertex_count(&molecule, None),
            vertex_count(&molecule, Some(&[true; 4])),
        );
    }

    #[test]
    fn hiding_everything_draws_nothing() {
        let molecule = chain(4);
        assert_eq!(vertex_count(&molecule, Some(&[false; 4])), 0);
    }

    /// A mask that has fallen out of step with the molecule should show too
    /// much rather than silently blank the view.
    #[test]
    fn a_short_mask_leaves_the_remaining_atoms_visible() {
        let molecule = chain(4);
        let (per_atom, per_bond) = costs(chain);

        // Only atom 0 is described, and it is hidden. Atoms 1..4 stay visible,
        // as do the two bonds between them.
        assert_eq!(
            vertex_count(&molecule, Some(&[false])),
            3 * per_atom + 2 * per_bond,
        );
    }
}
