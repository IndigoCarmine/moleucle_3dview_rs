//! A lightweight 3D molecule visualization library.
//!
//! This crate provides a `MoleculeViewer` struct for rendering molecule data loaded via `bio_files`.
//!
//! # Example
//!
//! ```no_run
//! use graphics::{run, Scene, UiSettings, GraphicsSettings, EngineUpdates, EntityUpdate, ControlScheme};
//! use lin_alg::f32::Vec3;
//! use moleucle_3dview_rs::{Molecule, MoleculeViewer, DebugRender, AdditionalRender};
//! use std::path::Path;
//!
//! fn main() {
//!     let mut viewer = MoleculeViewer::new();
//!     // viewer.set_molecule(Molecule::from_mol2(Path::new("Benzene.mol2")).unwrap());
//!
//!     let mut scene = Scene::default();
//!     scene.camera.position = Vec3::new(0.0, 0.0, -10.0);
//!
//!     viewer.update_scene(&mut scene);
//!    
//!     // ... standard graphics::run loop setup
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

pub use additional_render::{AdditionalRender, SelectedAtomRender, DebugRender};
pub use camera::{Camera, OrbitalCamera, ProjectionType};
pub use controller::CameraController;
pub use molecule::Molecule;
pub use offscreen_renderer::OffscreenRenderer;
pub use scene_types::{Entity, Mesh, Scene};
pub use ui::ViewerUiComponent;
pub use viewer::MoleculeViewer;
