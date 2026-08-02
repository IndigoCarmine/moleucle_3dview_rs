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
    /// Optional per-atom visibility mask (atom order); `None` shows everything.
    pub(super) visible: Option<&'a [bool]>,
}

impl StyleBuildContext<'_> {
    /// Whether atom `index` should be drawn.
    #[inline]
    pub(super) fn is_atom_visible(&self, index: usize) -> bool {
        crate::frame_state::is_visible(self.visible, index)
    }

    /// Whether both of a bond's endpoints are drawn. Hiding an atom hides the
    /// bonds hanging off it, or they would end in mid-air.
    #[inline]
    pub(super) fn is_bond_visible(&self, bond: &crate::molecule::Bond) -> bool {
        self.is_atom_visible(bond.atom_a) && self.is_atom_visible(bond.atom_b)
    }
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

/// The CPU vertex builder for `render_style`, or `None` for styles that are
/// drawn entirely by GPU instancing.
///
/// `Circles` has no CPU geometry — it is drawn as ray-traced sphere impostors
/// from an instance buffer — and the mesh styles fall back to that same
/// pipeline above `MAX_MESH_ATOMS`. Returning `None` rather than panicking
/// keeps a drift between that decision and this dispatch from taking the
/// process down.
pub(super) fn style_for(render_style: RenderStyle) -> Option<&'static dyn MolecularRenderStyle> {
    match render_style {
        RenderStyle::BallStick => Some(&ball_stick_style::BALL_STICK_STYLE),
        RenderStyle::BallOnly => Some(&ball_only_style::BALL_ONLY_STYLE),
        RenderStyle::Wireframe => Some(&wireframe_style::WIREFRAME_STYLE),
        RenderStyle::Circles => None,
    }
}
