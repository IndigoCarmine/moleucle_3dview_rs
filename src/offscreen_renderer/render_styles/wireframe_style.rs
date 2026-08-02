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
        context: &StyleBuildContext<'_>,
        molecule: Option<&Molecule>,
        color_fn: ColorFn,
    ) -> Vec<Vertex> {
        let max_vertices = MAX_RENDER_VERTICES;
        let opacity = context.molecule_opacity;
        let mut vertices = if let Some(mol) = molecule {
            let capacity = mol
                .bonds
                .len()
                .saturating_mul(4)
                .saturating_add(mol.atoms.len().saturating_mul(6))
                .saturating_add(6)
                .min(max_vertices);
            Vec::with_capacity(capacity)
        } else {
            Vec::with_capacity(6.min(max_vertices))
        };

        if let Some(mol) = molecule {
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
                        [0.70, 0.70, 0.72, opacity],
                        max_vertices,
                    ) {
                        break 'bonds;
                    }
                }
            }

            'atoms: for (i, atom) in mol.atoms.iter().enumerate() {
                if !context.is_atom_visible(i) {
                    continue;
                }
                let pos = atom.position;
                let span = 0.02;
                // Per-atom color override when supplied, else the color function.
                let base = context
                    .atom_colors
                    .and_then(|c| c.get(i).copied())
                    .unwrap_or_else(|| {
                        let c = color_fn(atom, false);
                        [c.0, c.1, c.2, c.3]
                    });
                let color = [base[0], base[1], base[2], base[3] * opacity];

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
            [1.0, 0.0, 0.0, 1.0],
            max_vertices,
        );
        let _ = append_line(
            &mut vertices,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, axis_len, 0.0),
            [0.0, 1.0, 0.0, 1.0],
            max_vertices,
        );
        let _ = append_line(
            &mut vertices,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, axis_len),
            [0.0, 0.0, 1.0, 1.0],
            max_vertices,
        );

        vertices
    }
}
