use crate::{
    camera,
    Camera,
    CameraController,
    Molecule,
    MoleculeViewer,
    offscreen_renderer::LodSettings,
    OffscreenRenderer,
    RenderStyle,
    SelectedAtomRender,
};
use eframe::egui::{self, PointerButton, Sense};

pub struct InteractiveMoleculeViewport {
    viewer: MoleculeViewer<SelectedAtomRender>,
    controller: CameraController<camera::OrbitalCamera>,
    offscreen: OffscreenRenderer,
}

impl InteractiveMoleculeViewport {
    pub fn new() -> Self {
        let mut viewer = MoleculeViewer::new();
        viewer.additional_render = Some(Box::new(SelectedAtomRender::new()));

        Self {
            viewer,
            controller: CameraController::<camera::OrbitalCamera>::new(),
            offscreen: OffscreenRenderer::new(),
        }
    }

    pub fn set_molecule(&mut self, molecule: Molecule) {
        self.viewer.set_molecule(molecule);
    }

    pub fn selected_atoms(&self) -> Vec<usize> {
        self.viewer.selected_atoms()
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
        ctx: &egui::Context,
        ui: &mut egui::Ui,
        render_state: &egui_wgpu::RenderState,
    ) -> Result<(), String> {
        let available = ui.available_size_before_wrap();
        let width = available.x.max(1.0) as u32;
        let height = available.y.max(1.0) as u32;
        self.controller.camera.set_aspect(width as f32 / height as f32);

        self.offscreen.ensure_resources(render_state, width, height)?;

        let selected = self.viewer.selected_atoms_ref();
        let view_proj = self.controller.camera.view_projection().data;
        self.offscreen.render_frame(
            render_state,
            self.viewer.molecule.as_ref(),
            selected,
            view_proj,
            self.viewer.color_fn,
        )?;

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
                    self.controller.camera.orbit(delta.x * 0.005, delta.y * 0.005);
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
                    let mut selected = self.viewer.selected_atoms();
                    if let Some(pos) = selected.iter().position(|&index| index == i) {
                        selected.remove(pos);
                    } else {
                        selected.push(i);
                    }
                    self.viewer.set_selected_atoms(selected);
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
