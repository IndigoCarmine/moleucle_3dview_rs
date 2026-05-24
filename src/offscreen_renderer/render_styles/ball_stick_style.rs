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
                let a = mol.atoms[bond.atom_a].position;
                let b = mol.atoms[bond.atom_b].position;
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
                            [0.55, 0.55, 0.55],
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
                        [0.55, 0.55, 0.55],
                        max_vertices,
                    ) {
                        break 'bonds;
                    }
                }
            }
        }

        'atoms: for atom in &mol.atoms {
            let pos = atom.position;
            let radius = ball_stick_radius(&atom.element, false);
            let color_tuple = color_fn(atom, false);
            let color = [color_tuple.0, color_tuple.1, color_tuple.2];

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

    if include_bonds {
        let axis_len = 0.2;
        let axis_radius = 0.01;
        if low_mode {
            let _ = append_line(
                &mut vertices,
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(axis_len, 0.0, 0.0),
                [1.0, 0.0, 0.0],
                max_vertices,
            );
            let _ = append_line(
                &mut vertices,
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, axis_len, 0.0),
                [0.0, 1.0, 0.0],
                max_vertices,
            );
            let _ = append_line(
                &mut vertices,
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(0.0, 0.0, axis_len),
                [0.0, 0.0, 1.0],
                max_vertices,
            );
        } else {
            let _ = append_mesh_triangles(
                &mut vertices,
                cylinder_mesh,
                Vec3::new(axis_len * 0.5, 0.0, 0.0),
                Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), -std::f32::consts::FRAC_PI_2),
                Vec3::new(axis_radius, axis_len, axis_radius),
                [1.0, 0.0, 0.0],
                max_vertices,
            );
            let _ = append_mesh_triangles(
                &mut vertices,
                cylinder_mesh,
                Vec3::new(0.0, axis_len * 0.5, 0.0),
                Quaternion::new_identity(),
                Vec3::new(axis_radius, axis_len, axis_radius),
                [0.0, 1.0, 0.0],
                max_vertices,
            );
            let _ = append_mesh_triangles(
                &mut vertices,
                cylinder_mesh,
                Vec3::new(0.0, 0.0, axis_len * 0.5),
                Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), std::f32::consts::FRAC_PI_2),
                Vec3::new(axis_radius, axis_len, axis_radius),
                [0.0, 0.0, 1.0],
                max_vertices,
            );
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
