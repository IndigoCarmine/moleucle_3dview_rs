//! A lightweight 3D molecule visualization library.
//!
//! This crate provides a `MoleculeViewer` struct for rendering molecule data in MOL2 and PDB formats.
//!
//! # Example
//!
//! ```no_run
//! use moleucle_3dview_rs::{Molecule, MoleculeViewer, SelectedAtomRender, ViewPortEvent};
//! use std::path::Path;
//!
//! fn main() {
//!     // Load a molecule from MOL2 or PDB file
//!     let mol = Molecule::from_mol2(Path::new("Benzene.mol2"))
//!         .or_else(|_| Molecule::from_pdb(Path::new("protein.pdb")))
//!         .expect("Failed to load molecule");
//!
//!     // Create a viewer
//!     let mut viewer: MoleculeViewer<SelectedAtomRender> = MoleculeViewer::new();
//!     viewer.set_molecule(mol);
//!
//!     // Access molecule data
//!     if let Some(molecule) = &viewer.molecule {
//!         println!("Atoms: {}", molecule.atoms.len());
//!         println!("Bonds: {}", molecule.bonds.len());
//!     }
//! }
//! ```

pub mod additional_render;
pub mod atom_radii;
pub mod camera;
pub mod controller;
pub mod frame_state;
pub mod molecule;
pub mod offscreen_renderer;
pub mod render_state;
pub mod scene_types;
pub mod viewer;
pub mod viewport;

pub use additional_render::{AdditionalRender, DebugRender, GpuPipeline, SelectedAtomRender};
pub use atom_radii::{ball_stick_radius, default_ball_stick_bond_radius, vdw_radius};
pub use camera::{Camera, OrbitalCamera, ProjectionType};
pub use controller::CameraController;
pub use frame_state::RenderFrameState;
pub use molecule::{Atom, AtomRecord, Molecule};
pub use offscreen_renderer::{LodSettings, OffscreenRenderer, RenderStyle};
pub use render_state::{
    get_state_clone_by_type, new_shared_states, set_state_by_type, with_state_mut_by_type,
    SharedRenderStates,
};
pub use scene_types::{Entity, Mesh, Scene};
pub use viewer::{default_color_fn, ColorFn, MoleculeViewer};
pub use viewport::{InteractiveMoleculeViewport, ViewPortEvent};
