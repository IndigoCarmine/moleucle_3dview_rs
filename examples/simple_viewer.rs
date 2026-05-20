use eframe::egui;
use moleucle_3dview_rs::additional_render::SelectedAtomRenderState;
// egui_wgpu::wgpu import removed (unused)
use moleucle_3dview_rs::frame_state::RenderFrameState;
use moleucle_3dview_rs::render_state::{get_state_clone_by_type, set_state_by_type};
use moleucle_3dview_rs::AdditionalRender;
use moleucle_3dview_rs::{InteractiveMoleculeViewport, Molecule, RenderStyle, ViewPortEvent};
use std::path::Path;

use std::cell::RefCell;
use std::rc::Rc;

struct SimpleViewerApp {
    viewport: InteractiveMoleculeViewport,
    render_state: Option<egui_wgpu::RenderState>,
    startup_error: Option<String>,
    hovered_atom: Rc<RefCell<usize>>,
    selected_atoms: Rc<RefCell<Vec<usize>>>,
}

#[derive(Clone)]
struct ExampleStateRender;

impl ExampleStateRender {
    fn new() -> Self {
        Self {}
    }
}

impl AdditionalRender for ExampleStateRender {
    fn update_scene(&self, scene: &mut moleucle_3dview_rs::Scene, frame: &RenderFrameState<'_>) {
        let Some(states) = frame.shared_states else {
            return;
        };

        // read a counter from state, default 0
        let count: usize = get_state_clone_by_type::<usize>(states).unwrap_or(0usize);

        // draw a small sphere whose color depends on count
        let color = match count % 3 {
            0 => (1.0, 0.0, 0.0),
            1 => (0.0, 1.0, 0.0),
            _ => (0.0, 0.0, 1.0),
        };

        // place sphere at top-left of scene for demo
        self.add_sphere_sameas_carbon(scene, lin_alg::f32::Vec3::new(0.0, 0.0, 0.0), 0.5, color);
        // increment and store counter for next frame
        set_state_by_type(states, count + 1usize);
    }
}

impl SimpleViewerApp {
    fn on_event(
        selected_atoms: Rc<RefCell<Vec<usize>>>,
        hovered_atom: Rc<RefCell<usize>>,
        viewport: &mut InteractiveMoleculeViewport,
        event: ViewPortEvent,
    ) {
        if let ViewPortEvent::clicked { atom } = event {
            // toggle atom selection in state
            selected_atoms.borrow_mut().push(atom);
            viewport.set_state_by_type(SelectedAtomRenderState {
                selected_atoms: selected_atoms.clone().borrow().clone(),
                color: [1.0, 0.0, 0.0],
            });
        }
        if let ViewPortEvent::hovered { atom } = event {
            *hovered_atom.borrow_mut() = atom;
        }
    }
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut viewport = InteractiveMoleculeViewport::new(None);
        // register an example per-render state renderer
        let example = ExampleStateRender::new();
        viewport.add_additional_render_box(Box::new(example));
        let startup_error = match load_default_molecule() {
            Ok(molecule) => {
                viewport.set_molecule(molecule);

                viewport.focus_on_molecule_center();
                None
            }
            Err(err) => Some(err),
        };

        Self {
            viewport,
            render_state: cc.wgpu_render_state.clone(),
            startup_error,
            hovered_atom: Rc::new(RefCell::new(0)),
            selected_atoms: Rc::new(RefCell::new(Vec::new())),
        }
    }

    fn register_event_handler(mut self) -> Self {
        let selected_atoms = Rc::clone(&self.selected_atoms);
        let hovered_atom = Rc::clone(&self.hovered_atom);

        self.viewport
            .register_event_handler(Box::new(move |viewport, event| {
                SimpleViewerApp::on_event(
                    Rc::clone(&selected_atoms),
                    Rc::clone(&hovered_atom),
                    viewport,
                    event,
                )
            }));
        self
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
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("help").show_inside(ui, |ui| {
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
        egui::Panel::bottom("footer").show_inside(ui, |ui| {
            ui.label("Hovered atom: ".to_string() + &self.hovered_atom.borrow().to_string())
        });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let Some(render_state) = &self.render_state else {
                ui.heading("WGPU backend is unavailable");
                ui.label("Start this example with the wgpu backend enabled in eframe.");
                return;
            };

            if let Err(err) = self.viewport.show(ui, render_state) {
                ui.colored_label(
                    egui::Color32::RED,
                    format!("Offscreen render failed: {err}"),
                );
            }
        });
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
        wgpu_options: Default::default(),
        ..Default::default()
    };

    eframe::run_native(
        "Molecule Viewer (Offscreen + egui)",
        options,
        Box::new(|cc| {
            let app = SimpleViewerApp::new(cc);
            Ok(Box::new(app.register_event_handler()))
        }),
    )
}
