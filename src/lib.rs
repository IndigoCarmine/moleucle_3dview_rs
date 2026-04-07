//! A lightweight 3D molecule visualization library.
//!
//! This crate provides a `MoleculeViewer` struct for rendering molecule data in MOL2 and PDB formats.
//!
//! # Example
//!
//! ```no_run
//! use moleucle_3dview_rs::{Molecule, MoleculeViewer, SelectedAtomRender};
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
pub mod camera;
pub mod controller;
pub mod molecule;
pub mod offscreen_renderer;
pub mod scene_types;
pub mod ui;
pub mod viewer;
pub mod viewport;

pub use additional_render::{AdditionalRender, SelectedAtomRender, DebugRender};
pub use camera::{Camera, OrbitalCamera, ProjectionType};
pub use controller::CameraController;
pub use molecule::{Molecule, Atom, AtomRecord};
pub use offscreen_renderer::{OffscreenRenderer, RenderStyle};
pub use scene_types::{Entity, Mesh, Scene};
pub use ui::ViewerUiComponent;
pub use viewer::{MoleculeViewer, ColorFn, default_color_fn};
pub use viewport::InteractiveMoleculeViewport;
