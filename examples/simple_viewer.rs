use eframe::egui;
use moleucle_3dview_rs::{
    InteractiveMoleculeViewport,
    Molecule,
    RenderStyle,
};
use std::path::Path;

struct SimpleViewerApp {
    viewport: InteractiveMoleculeViewport,
    render_state: Option<egui_wgpu::RenderState>,
}

impl SimpleViewerApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let mut viewport = InteractiveMoleculeViewport::new();
        load_default_molecule(&mut viewport);

        Self {
            viewport,
            render_state: cc.wgpu_render_state.clone(),
        }
    }
}

fn load_default_molecule(viewport: &mut InteractiveMoleculeViewport) {
    let path = Path::new("Benzene.mol2");
    if !path.exists() {
        eprintln!("Benzene.mol2 not found at {:?}", std::env::current_dir());
        return;
    }

    match Molecule::from_mol2(path) {
        Ok(mol) => {
            println!("Loaded molecule with {} atoms", mol.atoms.len());
            viewport.set_molecule(mol);
        }
        Err(_) => eprintln!("Failed to parse Benzene.mol2"),
    }
}

impl eframe::App for SimpleViewerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::TopBottomPanel::top("help").show(ctx, |ui| {
            ui.label("LMB: pick atom  RMB drag: orbit  MMB/Shift+RMB drag: pan  Wheel: dolly");
            ui.label(format!("Selected atoms: {:?}", self.viewport.selected_atoms()));
            ui.horizontal(|ui| {
                ui.label("Style:");
                let mut style = self.viewport.render_style();
                ui.selectable_value(&mut style, RenderStyle::BallStick, "BallStick");
                ui.selectable_value(&mut style, RenderStyle::Wireframe, "Wireframe");
                self.viewport.set_render_style(style);
            });
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
        viewport: egui::ViewportBuilder::default().with_inner_size([1280.0, 820.0]),
        ..Default::default()
    };

    eframe::run_native(
        "Molecule Viewer (Offscreen + egui)",
        options,
        Box::new(|cc| Ok(Box::new(SimpleViewerApp::new(cc)))),
    )
}
