use crate::viewer::ColorFn;
use crate::Molecule;

use super::super::VertexSink;
use super::ball_stick_style::emit_ballstick_vertices_into;
use super::{MolecularRenderStyle, StyleBuildContext};

pub(super) const BALL_ONLY_STYLE: BallOnlyStyle = BallOnlyStyle;

pub(super) struct BallOnlyStyle;

impl MolecularRenderStyle for BallOnlyStyle {
    fn primitive_stride(&self) -> usize {
        3
    }

    fn emit_vertices(
        &self,
        context: &StyleBuildContext<'_>,
        molecule: Option<&Molecule>,
        color_fn: ColorFn,
        sink: &mut dyn VertexSink,
    ) {
        emit_ballstick_vertices_into(context, molecule, color_fn, false, sink)
    }
}
