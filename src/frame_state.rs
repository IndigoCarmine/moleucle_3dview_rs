use crate::molecule::Molecule;
use crate::render_state::SharedRenderStates;
use crate::viewer::ColorFn;

pub struct RenderFrameState<'a> {
    pub molecule: Option<&'a Molecule>,
    pub view_proj: [f32; 16],
    pub color_fn: ColorFn,
    pub shared_states: Option<&'a SharedRenderStates>,
}

impl<'a> RenderFrameState<'a> {
    pub fn new(
        molecule: Option<&'a Molecule>,
        view_proj: [f32; 16],
        color_fn: ColorFn,
        shared_states: Option<&'a SharedRenderStates>,
    ) -> Self {
        Self {
            molecule,
            view_proj,
            color_fn,
            shared_states,
        }
    }
}
