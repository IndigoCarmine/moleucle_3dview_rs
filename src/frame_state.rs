use crate::molecule::Molecule;
use crate::render_state::SharedRenderStates;
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
}

impl<'a> RenderFrameState<'a> {
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
        }
    }
}
