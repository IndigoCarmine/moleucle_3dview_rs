use crate::{MoleculeViewer, SelectedAtomRender};
use egui::{Response, Ui, Widget};

pub struct ViewerUiWidget<'a> {
    component: &'a mut ViewerUiComponent,
    viewer: &'a mut MoleculeViewer<SelectedAtomRender>,
}

#[derive(Default)]
pub struct ViewerUiComponent {
    pub selection_input: String,
    pub last_status: String,
}

impl ViewerUiComponent {
    pub fn as_widget<'a>(
        &'a mut self,
        viewer: &'a mut MoleculeViewer<SelectedAtomRender>,
    ) -> ViewerUiWidget<'a> {
        ViewerUiWidget {
            component: self,
            viewer,
        }
    }

    pub fn draw_window(&mut self, ctx: &egui::Context, viewer: &mut MoleculeViewer<SelectedAtomRender>) {
        egui::Window::new("Molecule Viewer").show(ctx, |ui| {
            ui.add(self.as_widget(viewer));
        });
    }

    pub fn draw_contents(
        &mut self,
        ui: &mut egui::Ui,
        viewer: &mut MoleculeViewer<SelectedAtomRender>,
    ) -> Response {
        ui.vertical(|ui| {
        ui.label("Molecule Viewer");
        if let Some(mol) = &viewer.molecule {
            ui.label(format!("Atoms: {}", mol.atoms.len()));
            ui.label(format!("Bonds: {}", mol.bonds.len()));
        }

        ui.separator();
        ui.label("Controls:");
        ui.label("Right Click: Orbit");
        ui.label("Middle Click: Pan");
        ui.label("Scroll: Zoom");
        ui.label("Left Click: Select");

        ui.separator();
        ui.label("Selection I/O (comma-separated atom IDs)");
        ui.horizontal(|ui| {
            ui.text_edit_singleline(&mut self.selection_input);

            if ui.button("Read").clicked() {
                let selected = viewer.selected_atoms();
                self.selection_input = selected
                    .iter()
                    .map(|idx| idx.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                self.last_status = format!("Read {} selected atoms", selected.len());
            }

            if ui.button("Apply").clicked() {
                let atom_count = viewer.molecule.as_ref().map(|m| m.atoms.len()).unwrap_or(0);
                match parse_selection_input(&self.selection_input, atom_count) {
                    Ok(indices) => {
                        let count = indices.len();
                        viewer.set_selected_atoms(indices);
                        self.last_status = format!("Applied {count} selected atoms");
                    }
                    Err(e) => {
                        self.last_status = e;
                    }
                }
            }

            if ui.button("Clear").clicked() {
                viewer.clear_selected_atoms();
                self.selection_input.clear();
                self.last_status = "Selection cleared".to_string();
            }
        });

        let selected = viewer.selected_atoms();
        ui.label(format!("Current selection: {} atoms", selected.len()));
        if !selected.is_empty() {
            ui.label(format!("IDs: {:?}", selected));
        }
        if !self.last_status.is_empty() {
            ui.label(format!("Status: {}", self.last_status));
        }
        })
        .response
    }
}

impl Widget for ViewerUiWidget<'_> {
    fn ui(self, ui: &mut Ui) -> Response {
        self.component.draw_contents(ui, self.viewer)
    }
}

fn parse_selection_input(input: &str, atom_count: usize) -> Result<Vec<usize>, String> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Ok(Vec::new());
    }

    let mut out = Vec::new();
    for token in trimmed.split(',') {
        let token = token.trim();
        if token.is_empty() {
            continue;
        }

        let idx = token
            .parse::<usize>()
            .map_err(|_| format!("Invalid atom index: {token}"))?;
        if idx >= atom_count {
            return Err(format!("Atom index out of range: {idx} (max: {})", atom_count - 1));
        }
        out.push(idx);
    }

    out.sort_unstable();
    out.dedup();
    Ok(out)
}