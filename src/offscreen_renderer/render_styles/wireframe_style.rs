use lin_alg::f32::Vec3;

use crate::viewer::ColorFn;
use crate::Molecule;

use super::super::{append_line, bond_line_offsets, VertexSink};
use super::{MolecularRenderStyle, StyleBuildContext};

pub(super) const WIREFRAME_STYLE: WireframeStyle = WireframeStyle;

pub(super) struct WireframeStyle;

impl MolecularRenderStyle for WireframeStyle {
    fn primitive_stride(&self) -> usize {
        2
    }

    fn emit_vertices(
        &self,
        _context: &StyleBuildContext<'_>,
        molecule: Option<&Molecule>,
        color_fn: ColorFn,
        sink: &mut dyn VertexSink,
    ) {

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
                        sink,
                        a + off,
                        b + off,
                        [0.70, 0.70, 0.72],
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
                    sink,
                    pos + Vec3::new(-span, 0.0, 0.0),
                    pos + Vec3::new(span, 0.0, 0.0),
                    color,
                ) {
                    break 'atoms;
                }
                if !append_line(
                    sink,
                    pos + Vec3::new(0.0, -span, 0.0),
                    pos + Vec3::new(0.0, span, 0.0),
                    color,
                ) {
                    break 'atoms;
                }
                if !append_line(
                    sink,
                    pos + Vec3::new(0.0, 0.0, -span),
                    pos + Vec3::new(0.0, 0.0, span),
                    color,
                ) {
                    break 'atoms;
                }
            }
        }

        let axis_len = 2.0;
        let _ = append_line(
            sink,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(axis_len, 0.0, 0.0),
            [1.0, 0.0, 0.0],
        );
        let _ = append_line(
            sink,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, axis_len, 0.0),
            [0.0, 1.0, 0.0],
        );
        let _ = append_line(
            sink,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, axis_len),
            [0.0, 0.0, 1.0],
        );
    }
}
