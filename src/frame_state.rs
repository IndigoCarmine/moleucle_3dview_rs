use crate::molecule::Molecule;
use crate::render_state::SharedRenderStates;
use crate::offscreen_renderer::RenderStyle;
use crate::viewer::ColorFn;
use lin_alg::f32::Vec3;

pub struct RenderFrameState<'a> {
    pub molecule: Option<&'a Molecule>,
    pub view_proj: [f32; 16],
    pub camera_position: Option<Vec3>,
    pub fov_y: f32,
    pub camera_right: Vec3,
    pub camera_up: Vec3,
    pub camera_forward: Vec3,
    pub color_fn: ColorFn,
    pub shared_states: Option<&'a SharedRenderStates>,
    pub render_style: RenderStyle,
    pub mesh_resolution: usize,
    pub is_low_mode: bool,
    /// Opacity applied to the whole main molecule (atoms + bonds) in
    /// `0.0..=1.0`. Folded into each geometry color's alpha channel so the
    /// molecule can be faded without changing its `color_fn`. `1.0` is opaque.
    pub molecule_opacity: f32,
}

impl<'a> RenderFrameState<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        molecule: Option<&'a Molecule>,
        view_proj: [f32; 16],
        camera_position: Option<Vec3>,
        fov_y: f32,
        camera_right: Vec3,
        camera_up: Vec3,
        camera_forward: Vec3,
        color_fn: ColorFn,
        shared_states: Option<&'a SharedRenderStates>,
        render_style: RenderStyle,
        mesh_resolution: usize,
        is_low_mode: bool,
        molecule_opacity: f32,
    ) -> Self {
        Self {
            molecule,
            view_proj,
            camera_position,
            fov_y,
            camera_right,
            camera_up,
            camera_forward,
            color_fn,
            shared_states,
            render_style,
            mesh_resolution,
            is_low_mode,
            molecule_opacity,
        }
    }
}
