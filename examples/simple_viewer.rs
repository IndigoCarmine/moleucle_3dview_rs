use eframe::egui;
use moleucle_3dview_rs::{
    InteractiveMoleculeViewport,
    Molecule,
    RenderStyle,
};
use std::path::Path;
use egui_wgpu::{wgpu, WgpuSetup, WgpuSetupCreateNew};

struct SimpleViewerApp {
    viewport: InteractiveMoleculeViewport,
    render_state: Option<egui_wgpu::RenderState>,
    startup_error: Option<String>,
}

impl SimpleViewerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut viewport = InteractiveMoleculeViewport::new();
        let startup_error = match load_default_molecule() {
            Ok(molecule) => {
                viewport.set_molecule(molecule);
                None
            }
            Err(err) => Some(err),
        };

        Self {
            viewport,
            render_state: cc.wgpu_render_state.clone(),
            startup_error,
        }
    }
}

fn load_default_molecule() -> Result<Molecule, String> {
    let path = Path::new("A.pdb");
    if !path.exists() {
        return Err(format!(
            "A.pdb not found at {:?}",
            std::env::current_dir().map_err(|err| err.to_string())?
        ));
    }

    Molecule::from_pdb(path).map_err(|err| format!("Failed to parse A.pdb: {err}"))
}

impl eframe::App for SimpleViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("help").show(ctx, |ui| {
            ui.label("LMB: pick atom  RMB drag: orbit  MMB/Shift+RMB drag: pan  Wheel: dolly");
            ui.label(format!("Selected atoms: {}", self.viewport.selected_atoms().len()));
            if let Some(err) = &self.startup_error {
                ui.colored_label(egui::Color32::YELLOW, err);
            }
            ui.horizontal(|ui| {
                ui.label("Style:");
                let mut style = self.viewport.render_style();
                ui.selectable_value(&mut style, RenderStyle::BallStick, "BallStick");
                ui.selectable_value(&mut style, RenderStyle::Wireframe, "Wireframe");
                self.viewport.set_render_style(style);
            });

            ui.separator();
            ui.label("LOD settings (external config)");

            let mut lod = self.viewport.lod_settings();
            ui.checkbox(&mut lod.enabled, "Enable LOD auto optimization");
            ui.add(
                egui::Slider::new(&mut lod.high_detail_max_complexity, 10..=5000)
                    .text("High detail max complexity"),
            );
            ui.add(
                egui::Slider::new(&mut lod.medium_detail_max_complexity, 10..=10000)
                    .text("Medium detail max complexity"),
            );
            if lod.medium_detail_max_complexity < lod.high_detail_max_complexity {
                lod.medium_detail_max_complexity = lod.high_detail_max_complexity;
            }
            ui.add(
                egui::Slider::new(&mut lod.high_detail_mesh_resolution, 3..=32)
                    .text("High detail resolution"),
            );
            ui.add(
                egui::Slider::new(&mut lod.medium_detail_mesh_resolution, 3..=24)
                    .text("Medium detail resolution"),
            );
            ui.add(
                egui::Slider::new(&mut lod.low_detail_mesh_resolution, 3..=16)
                    .text("Low detail resolution"),
            );
            self.viewport.set_lod_settings(lod);

            if !lod.enabled {
                let mut resolution = self.viewport.mesh_resolution();
                ui.add(egui::Slider::new(&mut resolution, 3..=32).text("Manual mesh resolution"));
                self.viewport.set_mesh_resolution(resolution);
            } else {
                ui.label(format!(
                    "Auto mesh resolution: {}",
                    self.viewport.mesh_resolution()
                ));
            }
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            let Some(render_state) = &self.render_state else {
                ui.heading("WGPU backend is unavailable");
                ui.label("Start this example with the wgpu backend enabled in eframe.");
                return;
            };

            if let Err(err) = self.viewport.show(ctx, ui, render_state) {
                ui.colored_label(egui::Color32::RED, format!("Offscreen render failed: {err}"));
            }
        });

        ctx.request_repaint();
    }

    fn on_exit(&mut self) {
        if let Some(render_state) = &self.render_state {
            self.viewport.free_egui_texture(render_state);
        }
    }
}

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 820.0]),
        wgpu_options: egui_wgpu::WgpuConfiguration {
            wgpu_setup: WgpuSetup::CreateNew(WgpuSetupCreateNew {
                // Prefer Vulkan and fall back to DX12 when Vulkan runtime/driver is unavailable.
                instance_descriptor: wgpu::InstanceDescriptor {
                    backends: wgpu::Backends::VULKAN | wgpu::Backends::DX12,
                    ..Default::default()
                },
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            }),
            ..Default::default()
        },
        ..Default::default()
    };

    eframe::run_native(
        "Molecule Viewer (Offscreen + egui)",
        options,
        Box::new(|cc| Ok(Box::new(SimpleViewerApp::new(cc)))),
    )
}
