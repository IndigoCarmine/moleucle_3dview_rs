use lin_alg::f32::Vec3;

use crate::viewer::ColorFn;
use crate::Molecule;

use super::super::{append_line, bond_line_offsets, Vertex, MAX_RENDER_VERTICES};
use super::{MolecularRenderStyle, StyleBuildContext};

pub(super) const WIREFRAME_STYLE: WireframeStyle = WireframeStyle;

pub(super) struct WireframeStyle;

impl MolecularRenderStyle for WireframeStyle {
    fn primitive_stride(&self) -> usize {
        2
    }

    fn build_vertices(
        &self,
        _context: &StyleBuildContext<'_>,
        molecule: Option<&Molecule>,
        color_fn: ColorFn,
    ) -> Vec<Vertex> {
        let mut vertices = if let Some(mol) = molecule {
            let capacity = mol
                .bonds
                .len()
                .saturating_mul(4)
                .saturating_add(mol.atoms.len().saturating_mul(6))
                .saturating_add(6)
                .min(MAX_RENDER_VERTICES);
            Vec::with_capacity(capacity)
        } else {
            Vec::with_capacity(6.min(MAX_RENDER_VERTICES))
        };
        let max_vertices = usize::MAX;

        if let Some(mol) = molecule {
            'bonds: for bond in &mol.bonds {
                let a = mol.atoms[bond.atom_a].position;
                let b = mol.atoms[bond.atom_b].position;
                let diff = b - a;
                let len = diff.magnitude();
                if len < 0.001 {
                    continue;
                }

                let dir = diff.to_normalized();
                let mut lateral = Vec3::new(1.0, 0.0, 0.0);
                if dir.dot(lateral).abs() > 0.9 {
                    lateral = Vec3::new(0.0, 0.0, 1.0);
                }
                lateral = (lateral - dir * lateral.dot(dir)).to_normalized();

                let bond_order = bond.order.max(1) as usize;
                for offset in bond_line_offsets(bond_order) {
                    let off = lateral * offset;
                    if !append_line(
                        &mut vertices,
                        a + off,
                        b + off,
                        [0.70, 0.70, 0.72],
                        max_vertices,
                    ) {
                        break 'bonds;
                    }
                }
            }

            'atoms: for atom in &mol.atoms {
                let pos = atom.position;
                let span = 0.02;
                let color_tuple = color_fn(atom, false);
                let color = [color_tuple.0, color_tuple.1, color_tuple.2];

                if !append_line(
                    &mut vertices,
                    pos + Vec3::new(-span, 0.0, 0.0),
                    pos + Vec3::new(span, 0.0, 0.0),
                    color,
                    max_vertices,
                ) {
                    break 'atoms;
                }
                if !append_line(
                    &mut vertices,
                    pos + Vec3::new(0.0, -span, 0.0),
                    pos + Vec3::new(0.0, span, 0.0),
                    color,
                    max_vertices,
                ) {
                    break 'atoms;
                }
                if !append_line(
                    &mut vertices,
                    pos + Vec3::new(0.0, 0.0, -span),
                    pos + Vec3::new(0.0, 0.0, span),
                    color,
                    max_vertices,
                ) {
                    break 'atoms;
                }
            }
        }

        let axis_len = 2.0;
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

        vertices
    }
}
