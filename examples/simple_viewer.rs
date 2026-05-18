use eframe::egui;
use egui_wgpu::{wgpu, WgpuSetup, WgpuSetupCreateNew};
use moleucle_3dview_rs::render_state::{get_state_clone_by_type, set_state_by_type};
use moleucle_3dview_rs::AdditionalRender;
use moleucle_3dview_rs::{InteractiveMoleculeViewport, Molecule, RenderStyle};
use std::path::Path;

struct SimpleViewerApp {
    viewport: InteractiveMoleculeViewport,
    render_state: Option<egui_wgpu::RenderState>,
    startup_error: Option<String>,
}

#[derive(Clone)]
struct ExampleStateRender;

impl ExampleStateRender {
    fn new() -> Self {
        Self {}
    }
}

impl AdditionalRender for ExampleStateRender {
    fn update_scene(
        &self,
        scene: &mut moleucle_3dview_rs::Scene,
        _molecule: &moleucle_3dview_rs::Molecule,
        states: &moleucle_3dview_rs::SharedRenderStates,
    ) {
        // read a counter from state, default 0
        let count: usize = get_state_clone_by_type::<usize>(states).unwrap_or(0usize);

        // draw a small sphere whose color depends on count
        let color = match count % 3 {
            0 => (1.0, 0.0, 0.0),
            1 => (0.0, 1.0, 0.0),
            _ => (0.0, 0.0, 1.0),
        };

        // place sphere at top-left of scene for demo
        let pos = lin_alg::f32::Vec3::new(0.0, 0.0, 0.0);
        let sphere_mesh = moleucle_3dview_rs::Mesh::new_sphere(1.0, 8);
        let mesh_idx = scene.meshes.len();
        scene.meshes.push(sphere_mesh);
        let entity = moleucle_3dview_rs::Entity::new(
            mesh_idx,
            pos,
            lin_alg::f32::Quaternion::new_identity(),
            0.2,
            color,
            0.2,
        );
        scene.entities.push(entity);

        // increment and store counter for next frame
        set_state_by_type(states, count + 1usize);
    }
}

impl SimpleViewerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut viewport = InteractiveMoleculeViewport::new();
        // register an example per-render state renderer
        let example = ExampleStateRender::new();
        viewport.add_additional_render_box(Box::new(example));
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
            ui.label(format!(
                "Selected atoms: {}",
                self.viewport.selected_atoms().len()
            ));
            if let Some(err) = &self.startup_error {
                ui.colored_label(egui::Color32::YELLOW, err);
            }
            ui.horizontal(|ui| {
                ui.label("Style:");
                let mut style = self.viewport.render_style();
                ui.selectable_value(&mut style, RenderStyle::BallStick, "BallStick");
                ui.selectable_value(&mut style, RenderStyle::BallOnly, "BallOnly");
                ui.selectable_value(&mut style, RenderStyle::Wireframe, "Wireframe");
                self.viewport.set_render_style(style);
            });

            ui.separator();
            ui.label("LOD settings (distance-based, external config)");

            let mut lod = self.viewport.lod_settings();
            ui.checkbox(&mut lod.enabled, "Enable LOD auto optimization");
            ui.add(
                egui::Slider::new(&mut lod.distance_check_fps, 1.0..=60.0)
                    .text("Distance check FPS"),
            );
            ui.add(
                egui::Slider::new(&mut lod.high_detail_max_distance, 0.5..=50.0)
                    .text("High detail max distance"),
            );
            ui.add(
                egui::Slider::new(&mut lod.medium_detail_max_distance, 0.5..=100.0)
                    .text("Medium detail max distance"),
            );
            if lod.medium_detail_max_distance < lod.high_detail_max_distance {
                lod.medium_detail_max_distance = lod.high_detail_max_distance;
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
                ui.colored_label(
                    egui::Color32::RED,
                    format!("Offscreen render failed: {err}"),
                );
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
