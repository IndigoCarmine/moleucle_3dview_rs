use eframe::egui;
use lin_alg::f32::Vec3;
use moleucle_3dview_rs::additional_render::SelectedAtomRenderState;
// egui_wgpu::wgpu import removed (unused)
use moleucle_3dview_rs::frame_state::RenderFrameState;
use moleucle_3dview_rs::render_state::{get_state_clone_by_type, set_state_by_type};
use moleucle_3dview_rs::{default_color_fn, AdditionalRender, Scene};
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
    /// Per-atom colors from `default_color_fn`, kept so the per-atom alpha test
    /// can re-tint them without losing element coloring.
    base_atom_colors: Vec<[f32; 4]>,
    molecule_opacity: f32,
    atom_alpha_enabled: bool,
    atom_alpha: f32,
    /// Last alpha pushed through `set_atom_colors`. The renderer keys its
    /// geometry cache on the color slice's pointer + length, so re-uploading an
    /// identical Vec every frame would rebuild the mesh every frame.
    applied_atom_alpha: Option<f32>,
    probe: TransparencyProbeState,
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
            0 => (1.0, 0.0, 0.0, 1.0),
            1 => (0.0, 1.0, 0.0, 1.0),
            _ => (0.0, 0.0, 1.0, 1.0),
        };

        // place sphere at top-left of scene for demo
        self.add_sphere_sameas_carbon(
            scene,
            frame,
            lin_alg::f32::Vec3::new(0.0, 0.0, 0.0),
            0.5,
            color,
        );
        // increment and store counter for next frame
        set_state_by_type(states, count + 1usize);
    }
}

/// Shared state driving the transparency probes below.
#[derive(Clone, Copy, PartialEq)]
struct TransparencyProbeState {
    buried: bool,
    layers: bool,
    layer_alpha: f32,
}

impl Default for TransparencyProbeState {
    fn default() -> Self {
        Self {
            buried: true,
            layers: true,
            layer_alpha: 0.35,
        }
    }
}

/// Which of the three probe spheres a `TransparencyProbeRender` draws. Each
/// role is registered as its own `AdditionalRender`, so each becomes its own
/// draw batch and picks its own depth-write mode.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ProbeRole {
    /// Opaque sphere buried inside the molecule. Correctly hidden while the
    /// molecule is opaque; must become visible as `molecule_opacity` drops.
    Buried,
    /// Faded shell, registered *before* `LayerCore` so it draws first — the
    /// "dimmed inactive layer covers the active one" case.
    LayerShell,
    /// Opaque core sitting entirely inside `LayerShell`.
    LayerCore,
}

struct TransparencyProbeRender {
    role: ProbeRole,
}

/// Where the shell/core pair sits: just outside the molecule, so it stays
/// visible regardless of the molecule's own opacity.
fn layer_probe_center(center: Vec3, radius: f32) -> Vec3 {
    center + Vec3::new(0.0, radius * 1.5, 0.0)
}

impl AdditionalRender for TransparencyProbeRender {
    fn update_scene(&self, scene: &mut Scene, frame: &RenderFrameState<'_>) {
        let (Some(molecule), Some(states)) = (frame.molecule, frame.shared_states) else {
            return;
        };
        let probe = get_state_clone_by_type::<TransparencyProbeState>(states).unwrap_or_default();

        let center = molecule.center();
        let radius = molecule.radius().max(0.1);

        match self.role {
            ProbeRole::Buried if probe.buried => {
                self.add_sphere(scene, frame, center, radius * 0.45, (1.0, 0.4, 0.05, 1.0));
            }
            ProbeRole::LayerShell if probe.layers => {
                self.add_sphere(
                    scene,
                    frame,
                    layer_probe_center(center, radius),
                    radius * 0.40,
                    (0.25, 0.55, 1.0, probe.layer_alpha),
                );
            }
            ProbeRole::LayerCore if probe.layers => {
                self.add_sphere(
                    scene,
                    frame,
                    layer_probe_center(center, radius),
                    radius * 0.18,
                    (0.15, 1.0, 0.45, 1.0),
                );
            }
            _ => {}
        }
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
            let mut selected = selected_atoms.borrow_mut();
            if let Some(pos) = selected.iter().position(|&index| index == atom) {
                selected.remove(pos);
            } else {
                selected.push(atom);
            }
            viewport.set_state_by_type(SelectedAtomRenderState {
                selected_atoms: selected.clone(),
                color: [1.0, 0.0, 0.0, 1.0],
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

        // Transparency probes. Registration order is the draw order within the
        // pass, so LayerShell must come before LayerCore for that test to mean
        // anything.
        for role in [
            ProbeRole::Buried,
            ProbeRole::LayerShell,
            ProbeRole::LayerCore,
        ] {
            viewport.add_additional_render_box(Box::new(TransparencyProbeRender { role }));
        }

        let mut base_atom_colors = Vec::new();
        let startup_error = match load_default_molecule() {
            Ok(molecule) => {
                base_atom_colors = molecule
                    .atoms
                    .iter()
                    .map(|atom| {
                        let (r, g, b, a) = default_color_fn(atom, false);
                        [r, g, b, a]
                    })
                    .collect();
                viewport.set_molecule(molecule);

                viewport.focus_on_molecule_center();
                None
            }
            Err(err) => Some(err),
        };

        let probe = TransparencyProbeState::default();
        viewport.set_state_by_type(probe);

        Self {
            molecule_opacity: viewport.molecule_opacity(),
            viewport,
            render_state: cc.wgpu_render_state.clone(),
            startup_error,
            hovered_atom: Rc::new(RefCell::new(0)),
            selected_atoms: Rc::new(RefCell::new(Vec::new())),
            base_atom_colors,
            atom_alpha_enabled: false,
            atom_alpha: 0.35,
            applied_atom_alpha: None,
            probe,
        }
    }

    /// The transparency-test controls. Everything here exercises the
    /// depth-write-vs-translucency path in the offscreen renderer.
    fn transparency_controls(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Molecule opacity:");
            if ui
                .add(egui::Slider::new(&mut self.molecule_opacity, 0.0..=1.0))
                .changed()
            {
                self.viewport.set_molecule_opacity(self.molecule_opacity);
            }
            if ui.button("Reset to 1.0").clicked() {
                self.molecule_opacity = 1.0;
                self.viewport.set_molecule_opacity(1.0);
            }
        });

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.atom_alpha_enabled, "Per-atom alpha");
            ui.add_enabled(
                self.atom_alpha_enabled,
                egui::Slider::new(&mut self.atom_alpha, 0.0..=1.0),
            );
            ui.label("(set_atom_colors path, independent of the opacity slider)");
        });

        // Only push a new Vec when the value actually changes — see
        // `applied_atom_alpha`.
        let desired_alpha = self.atom_alpha_enabled.then_some(self.atom_alpha);
        if desired_alpha != self.applied_atom_alpha {
            self.applied_atom_alpha = desired_alpha;
            let colors = desired_alpha.map(|alpha| {
                self.base_atom_colors
                    .iter()
                    .map(|c| [c[0], c[1], c[2], alpha])
                    .collect::<Vec<_>>()
            });
            self.viewport.set_atom_colors(colors);
        }

        ui.horizontal(|ui| {
            ui.checkbox(&mut self.probe.buried, "Buried probe (orange, at center)");
            ui.checkbox(&mut self.probe.layers, "Layer probe (above molecule)");
            ui.label("shell alpha:");
            ui.add_enabled(
                self.probe.layers,
                egui::Slider::new(&mut self.probe.layer_alpha, 0.0..=1.0),
            );
        });
        self.viewport.set_state_by_type(self.probe);

        ui.label(
            "Expected: at opacity 1.0 the orange sphere is hidden. Lower the opacity and it \
             (plus the molecule's interior atoms) fades into view. The green core inside the \
             blue shell is visible whenever the shell is translucent, and disappears behind \
             the shell at alpha 1.0. Try every render style — each uses a different pipeline.",
        );
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
    // Allow `cargo run --example simple_viewer -- <file>`; otherwise prefer the
    // sample GROMACS trajectory frame if present, then fall back to A.pdb.
    let arg = std::env::args().nth(1);
    let candidates: Vec<&str> = match arg.as_deref() {
        Some(path) => vec![path],
        None => vec!["output.gro", "A.pdb"],
    };

    let path = candidates
        .iter()
        .map(Path::new)
        .find(|p| p.exists())
        .ok_or_else(|| {
            format!(
                "no molecule file found (tried {:?}) in {:?}",
                candidates,
                std::env::current_dir().unwrap_or_default()
            )
        })?;

    Molecule::load(path).map_err(|err| format!("Failed to parse {}: {err}", path.display()))
}

impl eframe::App for SimpleViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        egui::Panel::top("help").show(ui, |ui| {
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
                ui.selectable_value(&mut style, RenderStyle::Circles, "Circles");
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
        egui::Panel::bottom("footer").show(ui, |ui| {
            ui.label("Hovered atom: ".to_string() + &self.hovered_atom.borrow().to_string());
            ui.separator();
            ui.heading("Transparency / depth-write test");
            self.transparency_controls(ui);
        });

        egui::CentralPanel::default().show(ui, |ui| {
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
