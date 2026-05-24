use crate::viewer::ColorFn;
use crate::Molecule;

use super::super::Vertex;
use super::ball_stick_style::build_ballstick_vertices;
use super::{MolecularRenderStyle, StyleBuildContext};

pub(super) const BALL_ONLY_STYLE: BallOnlyStyle = BallOnlyStyle;

pub(super) struct BallOnlyStyle;

impl MolecularRenderStyle for BallOnlyStyle {
    fn primitive_stride(&self) -> usize {
        3
    }

    fn build_vertices(
        &self,
        context: &StyleBuildContext<'_>,
        molecule: Option<&Molecule>,
        color_fn: ColorFn,
    ) -> Vec<Vertex> {
        build_ballstick_vertices(context, molecule, color_fn, false)
    }
}
