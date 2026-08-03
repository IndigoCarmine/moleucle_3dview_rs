# moleucle_3dview_rs

A lightweight 3D molecule viewer for [egui](https://crates.io/crates/egui), rendering
through `wgpu`.

The crate draws into its own off-screen color/depth textures with hand-written wgpu
pipelines and inline WGSL, then hands the result to egui as a texture — so the 3D view
is just another widget inside an ordinary egui layout, not a separate window.

## Features

- **Four render styles**: ball-and-stick, ball-only, wireframe, and ray-traced sphere
  impostors (`Circles`).
- **Scales to MD-sized systems**: geometry is cached and only rebuilt when an input
  actually changes; above ~58k atoms the mesh styles fall back automatically to the
  instanced impostor pipeline, with bonds drawn as instanced cylinders. Systems of
  several million atoms render.
- **Trajectory playback**: `update_positions` moves atoms in place, keeping bonds,
  metadata and the camera, and re-uploads only the position-dependent buffers.
- **Periodic boundary conditions**: `set_periodic_images` tiles the molecule
  across its simulation cell, per-axis, for rectangular and triclinic boxes
  alike. The geometry is drawn once per image rather than duplicated, so a
  replica costs one draw and 16 bytes; off-screen images are frustum-culled, and
  picking a replica selects the atom it is a copy of.
- **Partial visibility**: `set_visible_atoms` hides atoms and their bonds
  without renumbering anything, so a host showing a subset does not have to
  maintain a parallel index space.
- **Level of detail**: an optional background worker lowers mesh resolution with
  camera distance (`LodSettings`).
- **Transparency**: per-molecule opacity and per-atom RGBA, drawn in a two-phase
  opaque-then-translucent pass.
- **Picking**: CPU ray picking for atoms and bonds, surfaced as viewport events.
- **Overlays**: implement `AdditionalRender` to draw your own geometry in the same
  pass — spheres, cylinders and raw meshes, on the triangle, wireframe or impostor
  pipeline. Ships with a selected-atom highlight and a debug-ray overlay.
- **Image export**: `render_image` renders off-screen at an arbitrary size with
  optional region cropping and supersampling, independent of the on-screen viewport.
- **File formats**: `.mol2`, `.pdb` and GROMACS `.gro` loaders, plus `AtomRecord`
  for reading and writing PDB `ATOM`/`HETATM` lines.

## Units

**Every length in the public API is in nanometers** — positions, radii, cell edges.
Loaders convert on the way in (PDB and MOL2 store Ångström; GRO already stores
nanometers). The conversion factor is exported as `ANGSTROM_TO_NM` for callers writing
their own parsers, and the `_angstrom` position setters are the only entry points that
take Ångström.

## Usage

`InteractiveMoleculeViewport` is the type applications use: it owns the camera, the
mouse handling, the off-screen renderer and the overlay list.

```rust,no_run
use moleucle_3dview_rs::{InteractiveMoleculeViewport, Molecule, SelectedAtomRender};
use std::path::Path;

struct App {
    viewport: InteractiveMoleculeViewport,
}

impl App {
    fn new(path: &Path) -> Result<Self, String> {
        let mut viewport = InteractiveMoleculeViewport::new();
        viewport.add_additional_render_box(Box::new(SelectedAtomRender::new()));

        viewport.set_molecule(Molecule::load(path)?);
        viewport.focus_on_molecule_center();

        Ok(Self { viewport })
    }

    // Call once per frame from inside an egui layout. `render_state` comes from
    // `eframe::CreationContext::wgpu_render_state`.
    fn ui(&mut self, ui: &mut egui::Ui, render_state: &egui_wgpu::RenderState) {
        if let Err(err) = self.viewport.show(ui, render_state) {
            ui.colored_label(egui::Color32::RED, err);
        }
    }
}
```

eframe must be running on the wgpu backend (`eframe::Renderer::Wgpu`), and
`viewport.free_egui_texture(render_state)` should be called from `App::on_exit`.

`MoleculeViewer` is the lower-level piece the viewport is built on — the molecule, its
appearance overrides and CPU ray picking, with no egui or camera involvement. Reach for
it only when driving `OffscreenRenderer` directly.

## Running the example

`examples/simple_viewer.rs` is a complete eframe app with a style picker, LOD sliders
and transparency controls.

```bash
cargo run --example simple_viewer            # loads A.pdb from the crate root
cargo run --example simple_viewer -- FILE    # or any .mol2 / .pdb / .gro
```

## License

MIT
