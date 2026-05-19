use crate::additional_render::{SelectedAtomRender, SelectedAtomRenderState};
use crate::frame_state::RenderFrameState;

use crate::render_state::{
    get_state_clone_by_type, new_shared_states, set_state_by_type, SharedRenderStates,
};
use crate::{
    camera, offscreen_renderer::LodSettings, Camera, CameraController, Molecule, MoleculeViewer,
    OffscreenRenderer, RenderStyle,
};
use eframe::egui::{self, PointerButton, Sense};

pub struct InteractiveMoleculeViewport {
    viewer: MoleculeViewer,
    controller: CameraController<camera::OrbitalCamera>,
    offscreen: OffscreenRenderer,
    shared_states: SharedRenderStates,
}

impl InteractiveMoleculeViewport {
    pub fn new() -> Self {
        let viewer = MoleculeViewer::new();
        let shared_states = new_shared_states();
        let mut offscreen = OffscreenRenderer::new();
        offscreen.add_additional_render(Box::new(SelectedAtomRender::new()));

        Self {
            viewer,
            controller: CameraController::<camera::OrbitalCamera>::new(),
            offscreen: offscreen,
            shared_states,
        }
    }

    /// Expose a way to add an AdditionalRender to the internal viewer from callers.
    pub fn add_additional_render_box(&mut self, render: Box<dyn crate::AdditionalRender>) {
        self.offscreen.add_additional_render(render);
    }

    pub fn set_molecule(&mut self, molecule: Molecule) {
        self.viewer.set_molecule(molecule);
    }

    pub fn selected_atoms(&self) -> Vec<usize> {
        get_state_clone_by_type::<SelectedAtomRenderState>(&self.shared_states)
            .map(|state| state.selected_atoms)
            .unwrap_or_default()
    }
    pub fn focus_on_molecule_center(&mut self) {
        if let Some(molecule) = self.viewer.molecule.as_ref() {
            self.controller.camera.center = molecule.center();
            self.controller.camera.radius = molecule.radius() * 2.0;
        }
    }

    pub fn render_style(&self) -> RenderStyle {
        self.offscreen.render_style()
    }

    pub fn mesh_resolution(&self) -> usize {
        self.offscreen.mesh_resolution()
    }

    pub fn set_mesh_resolution(&mut self, mesh_resolution: usize) {
        self.offscreen.set_mesh_resolution(mesh_resolution);
    }

    pub fn lod_settings(&self) -> LodSettings {
        self.offscreen.lod_settings()
    }

    pub fn set_lod_settings(&mut self, lod_settings: LodSettings) {
        self.offscreen.set_lod_settings(lod_settings);
    }

    pub fn set_render_style(&mut self, render_style: RenderStyle) {
        self.offscreen.set_render_style(render_style);
    }

    pub fn free_egui_texture(&mut self, render_state: &egui_wgpu::RenderState) {
        self.offscreen.free_egui_texture(render_state);
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        render_state: &egui_wgpu::RenderState,
    ) -> Result<(), String> {
        let available = ui.available_size_before_wrap();
        let width = available.x.max(1.0) as u32;
        let height = available.y.max(1.0) as u32;
        self.controller
            .camera
            .set_aspect(width as f32 / height as f32);

        if let Some(molecule) = self.viewer.molecule.as_ref() {
            let camera_position = self.controller.camera.position();
            let distance = (camera_position - molecule.center()).magnitude();
            self.offscreen.submit_lod_distance(distance);
        }

        self.offscreen
            .ensure_resources(render_state, width, height)?;

        let view_proj = self.controller.camera.view_projection().data;
        let frame = RenderFrameState::new(
            self.viewer.molecule.as_ref(),
            view_proj,
            self.viewer.color_fn,
            Some(&self.shared_states),
        );

        self.offscreen
            .render_frame_with_state(render_state, &frame)?;

        let texture_id = self
            .offscreen
            .texture_id()
            .ok_or_else(|| "No texture id registered".to_string())?;

        let response = ui.add(
            egui::Image::from_texture(egui::load::SizedTexture::new(
                texture_id,
                egui::vec2(width as f32, height as f32),
            ))
            .sense(Sense::click_and_drag()),
        );

        let ctx = ui.ctx();
        self.handle_interaction(ctx, &response);
        Ok(())
    }

    fn handle_interaction(&mut self, ctx: &egui::Context, response: &egui::Response) {
        if response.hovered() {
            let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > f32::EPSILON {
                self.controller.camera.dolly(scroll * 0.02);
            }
        }

        if response.hovered() {
            let (delta, sec_down, mid_down, shift_down) = ctx.input(|i| {
                (
                    i.pointer.delta(),
                    i.pointer.button_down(PointerButton::Secondary),
                    i.pointer.button_down(PointerButton::Middle),
                    i.modifiers.shift,
                )
            });

            // Use raw pointer delta so secondary/middle drag works reliably on the image widget.
            if sec_down || mid_down {
                if mid_down || shift_down {
                    self.controller
                        .camera
                        .pan(lin_alg::f32::Vec2::new(delta.x * 0.01, delta.y * 0.01));
                } else {
                    self.controller
                        .camera
                        .orbit(delta.x * 0.005, delta.y * 0.005);
                }
            }
        }

        if response.clicked_by(PointerButton::Primary) {
            if let Some(pointer) = response.interact_pointer_pos() {
                let local = pointer - response.rect.min;
                let (ray_origin, ray_dir) = self.controller.camera.ray_from_screen(
                    local.x,
                    local.y,
                    response.rect.width().max(1.0),
                    response.rect.height().max(1.0),
                );

                if let Some(crate::viewer::ViewerEvent::AtomClicked(i)) =
                    self.viewer.pick(ray_origin, ray_dir)
                {
                    let mut selected: SelectedAtomRenderState =
                        get_state_clone_by_type::<SelectedAtomRenderState>(&self.shared_states)
                            .unwrap_or_else(|| SelectedAtomRenderState {
                                selected_atoms: Vec::new(),
                                color: [1.0, 0.0, 0.0],
                            });
                    if let Some(_) = selected.selected_atoms.iter().position(|&index| index == i) {
                        selected.remove_atom(i);
                    } else {
                        selected.toggle_atom(i);
                    }
                    set_state_by_type(&self.shared_states, selected);
                }
            }
        }
    }
}

impl Default for InteractiveMoleculeViewport {
    fn default() -> Self {
        Self::new()
    }
}
