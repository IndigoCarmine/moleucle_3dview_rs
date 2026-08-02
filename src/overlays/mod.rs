//! Ready-made [`crate::AdditionalRender`] implementations.
//!
//! Each overlay reads its configuration from the shared render-state map
//! (`viewport.set_state_by_type(...)`) rather than from constructor arguments,
//! so an application registers the overlay once at startup and then drives it by
//! pushing state. An overlay whose state has never been set draws nothing —
//! except [`AxesRender`], which defaults to visible.
//!
//! These are primitives, not features: `SphereSetRender` draws whatever spheres
//! you hand it, `AtomGroupRender` colours whatever index groups you hand it. The
//! meaning of the groups — index files, selections, chains — stays in the
//! application.

pub mod atom_group;
pub mod atom_pair;
pub mod axes;
pub mod debug_ray;
pub mod point_cloud;
pub mod selected_atom;
pub mod simulation_cell;
pub mod sphere_set;

pub use atom_group::{AtomGroup, AtomGroupRender, AtomGroupState};
pub use atom_pair::{AtomPairRender, AtomPairState};
pub use axes::{AxesRender, AxesState};
pub use debug_ray::{DebugRender, DebugRenderState};
pub use point_cloud::{PointCloudLayer, PointCloudRender, PointCloudState};
pub use selected_atom::{SelectedAtomRender, SelectedAtomRenderState};
pub use simulation_cell::{SimulationCellRender, SimulationCellState};
pub use sphere_set::{OverlaySphere, SphereSet, SphereSetRender, SphereSetState};
