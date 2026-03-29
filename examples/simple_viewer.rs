use eframe::egui::{self, PointerButton, Sense};
use moleucle_3dview_rs::{
    camera,
    Camera,
    CameraController,
    Molecule,
    MoleculeViewer,
    OffscreenRenderer,
    SelectedAtomRender,
};
use std::path::Path;

struct SimpleViewerApp {
    viewer: MoleculeViewer<SelectedAtomRender>,
    controller: CameraController<camera::OrbitalCamera>,
    offscreen: OffscreenRenderer,
    render_state: Option<egui_wgpu::RenderState>,
}

impl SimpleViewerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut viewer = MoleculeViewer::new();
        load_default_molecule(&mut viewer);
        viewer.additional_render = Some(Box::new(SelectedAtomRender::new()));

        Self {
            viewer,
            controller: CameraController::<camera::OrbitalCamera>::new(),
            offscreen: OffscreenRenderer::new(),
            render_state: cc.wgpu_render_state.clone(),
        }
    }
}

fn load_default_molecule(viewer: &mut MoleculeViewer<SelectedAtomRender>) {
    let path = Path::new("Benzene.mol2");
    if !path.exists() {
        eprintln!("Benzene.mol2 not found at {:?}", std::env::current_dir());
        return;
    }

    match Molecule::from_mol2(path) {
        Ok(mol) => {
            println!("Loaded molecule with {} atoms", mol.atoms.len());
            viewer.set_molecule(mol);
        }
        Err(_) => eprintln!("Failed to parse Benzene.mol2"),
    }
}

impl eframe::App for SimpleViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("help").show(ctx, |ui| {
            ui.label("LMB: pick atom  RMB drag: orbit  MMB/Shift+RMB drag: pan  Wheel: dolly");
            ui.label(format!("Selected atoms: {:?}", self.viewer.selected_atoms()));
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(render_state) = &self.render_state else {
                ui.heading("WGPU backend is unavailable");
                ui.label("Start this example with the wgpu backend enabled in eframe.");
                return;
            };

            let available = ui.available_size_before_wrap();
            let width = available.x.max(1.0) as u32;
            let height = available.y.max(1.0) as u32;
            self.controller.camera.set_aspect(width as f32 / height as f32);

            if let Err(err) = self.offscreen.ensure_resources(render_state, width, height) {
                ui.colored_label(egui::Color32::RED, format!("Offscreen init failed: {err}"));
                return;
            }

            let selected = self.viewer.selected_atoms();
            let view_proj = self.controller.camera.view_projection().data;
            if let Err(err) = self.offscreen.render_frame(
                render_state,
                self.viewer.molecule.as_ref(),
                &selected,
                view_proj,
            ) {
                ui.colored_label(egui::Color32::RED, format!("Offscreen render failed: {err}"));
                return;
            }

            let Some(texture_id) = self.offscreen.texture_id() else {
                ui.colored_label(egui::Color32::RED, "No texture id registered");
                return;
            };

            let response = ui.add(
                egui::Image::from_texture(egui::load::SizedTexture::new(
                    texture_id,
                    egui::vec2(width as f32, height as f32),
                ))
                .sense(Sense::click_and_drag()),
            );

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

                    if let Some(event) = self.viewer.pick(ray_origin, ray_dir) {
                        if let moleucle_3dview_rs::viewer::ViewerEvent::AtomClicked(i) = event {
                            if let Some(selected_atom) = &mut self.viewer.additional_render {
                                selected_atom.toggle_atom(i);
                                self.viewer.dirty = true;
                            }
                        }
                    }
                }
            }
        });

        ctx.request_repaint();
    }

    fn on_exit(&mut self) {
        if let Some(render_state) = &self.render_state {
            self.offscreen.free_egui_texture(render_state);
        }
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 820.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Molecule Viewer (Offscreen + egui)",
        options,
        Box::new(|cc| Ok(Box::new(SimpleViewerApp::new(cc)))),
    )
}
