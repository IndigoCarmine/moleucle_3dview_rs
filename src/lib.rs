//! A lightweight 3D molecule viewer for [`egui`], rendering through `wgpu`.
//!
//! The crate draws into its own off-screen color/depth textures and hands the
//! result to egui as a texture, so the 3D view is just another widget inside an
//! ordinary egui layout.
//!
//! # Units
//!
//! **Every length in this crate's public API is in nanometers** — atom
//! positions, radii, bond radii, simulation cell edges. Loaders convert on the
//! way in (PDB and MOL2 store Ångström; GRO already stores nanometers), and the
//! Å→nm factor is exposed as [`ANGSTROM_TO_NM`] for callers writing their own
//! parsers. The `_angstrom` variants of the position setters are the only
//! entry points that take Ångström.
//!
//! # Entry point
//!
//! [`InteractiveMoleculeViewport`] is the type applications use: it owns the
//! camera, the mouse handling, the off-screen renderer and the overlay list.
//!
//! ```no_run
//! # use moleucle_3dview_rs::{InteractiveMoleculeViewport, Molecule, SelectedAtomRender};
//! # use std::path::Path;
//! # fn demo(ui: &mut egui::Ui, render_state: &egui_wgpu::RenderState) -> Result<(), String> {
//! let mut viewport = InteractiveMoleculeViewport::new(None);
//! viewport.add_additional_render_box(Box::new(SelectedAtomRender::new()));
//!
//! let molecule = Molecule::load(Path::new("Benzene.mol2"))?;
//! viewport.set_molecule(molecule);
//! viewport.focus_on_molecule_center();
//!
//! // Once per frame, from inside an egui layout:
//! viewport.show(ui, render_state)?;
//! # Ok(())
//! # }
//! ```
//!
//! [`MoleculeViewer`] is the lower-level piece the viewport is built on — the
//! molecule, its appearance overrides and CPU ray picking, with no egui or
//! camera involvement. Reach for it only when driving [`OffscreenRenderer`]
//! directly.

/// Ångström → nanometer conversion factor.
///
/// This crate works in nanometers throughout (see the crate-level docs). File
/// formats that store Ångström are converted by this factor on load; callers
/// writing their own parsers should apply it themselves rather than
/// re-declaring the constant.
pub const ANGSTROM_TO_NM: f32 = 0.1;

/// Nanometer → Ångström conversion factor, the inverse of [`ANGSTROM_TO_NM`].
///
/// Useful when writing this crate's nanometer coordinates back out to a format
/// that stores Ångström (PDB, MOL2).
pub const NM_TO_ANGSTROM: f32 = 10.0;

pub mod additional_render;
pub mod atom_radii;
pub mod camera;
pub mod frame_state;
pub mod molecule;
pub mod offscreen_renderer;
pub mod overlays;
pub mod render_state;
pub mod scene_types;
pub mod viewer;
pub mod viewport;

pub use additional_render::{AdditionalRender, GpuPipeline};
pub use atom_radii::{ball_stick_radius, default_ball_stick_bond_radius, vdw_radius};
pub use camera::{Camera, OrbitalCamera};
pub use frame_state::{RenderFrameState, DEFAULT_CLEAR_COLOR};
pub use molecule::{Atom, AtomMeta, AtomRecord, Bond, Element, Molecule};
pub use offscreen_renderer::{LodSettings, OffscreenRenderer, RenderStyle};
pub use overlays::{
    AtomGroup, AtomGroupRender, AtomGroupState, AtomPairRender, AtomPairState, AxesRender,
    AxesState, DebugRender, DebugRenderState, OverlaySphere, PointCloudLayer, PointCloudRender,
    PointCloudState, SelectedAtomRender, SelectedAtomRenderState, SimulationCellRender,
    SimulationCellState, SphereSet, SphereSetRender, SphereSetState,
};
pub use render_state::{
    get_state_clone_by_type, new_shared_states, set_state_by_type, with_state_by_type,
    with_state_mut_by_type, SharedRenderStates,
};
pub use scene_types::{Entity, Mesh, Scene};
pub use viewer::{default_color_fn, ColorFn, MoleculeViewer};
pub use viewport::{
    ExportedImage, ImageExportRequest, InteractiveMoleculeViewport, ViewPortEvent,
};
