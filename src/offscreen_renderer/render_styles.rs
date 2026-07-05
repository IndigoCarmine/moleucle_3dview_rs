mod ball_only_style;
mod ball_stick_style;
pub(super) mod circles;
mod wireframe_style;

use crate::viewer::ColorFn;
use crate::Molecule;

use super::{OffscreenRendererPreference, RenderMesh, RenderStyle, Vertex};

pub(super) struct StyleBuildContext<'a> {
    pub(super) preference: OffscreenRendererPreference,
    pub(super) sphere_mesh: &'a RenderMesh,
    pub(super) cylinder_mesh: &'a RenderMesh,
    /// Whole-molecule opacity in `0.0..=1.0`, folded into atom/bond alpha.
    pub(super) molecule_opacity: f32,
    /// Optional per-atom radius / RGBA color overrides (atom order); `None`
    /// falls back to element-derived radii and `color_fn`.
    pub(super) atom_radii: Option<&'a [f32]>,
    pub(super) atom_colors: Option<&'a [[f32; 4]]>,
}

pub(super) trait MolecularRenderStyle {
    fn primitive_stride(&self) -> usize;

    fn build_vertices(
        &self,
        context: &StyleBuildContext<'_>,
        molecule: Option<&Molecule>,
        color_fn: ColorFn,
    ) -> Vec<Vertex>;
}

pub(super) fn style_for(render_style: RenderStyle) -> &'static dyn MolecularRenderStyle {
    match render_style {
        RenderStyle::BallStick => &ball_stick_style::BALL_STICK_STYLE,
        RenderStyle::BallOnly => &ball_only_style::BALL_ONLY_STYLE,
        RenderStyle::Wireframe => &wireframe_style::WIREFRAME_STYLE,
        RenderStyle::Circles => {
            panic!("Circles style uses a dedicated GPU instancing pipeline")
        }
    }
}
