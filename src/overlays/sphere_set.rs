//! Arbitrary sets of spheres, independent of the loaded molecule.

use crate::frame_state::RenderFrameState;
use crate::render_state::with_state_by_type;
use crate::scene_types::Scene;
use crate::{AdditionalRender, GpuPipeline};
use lin_alg::f32::Vec3;

/// One sphere: position and radius in nanometers, plus straight RGBA.
#[derive(Clone, Copy, Debug)]
pub struct OverlaySphere {
    pub position: Vec3,
    pub radius: f32,
    pub color: (f32, f32, f32, f32),
}

/// A group of spheres the application manages as a unit.
#[derive(Clone, Debug, Default)]
pub struct SphereSet {
    pub spheres: Vec<OverlaySphere>,
}

/// Sphere sets to draw.
///
/// The viewer renders bonds and answers picks for exactly one molecule, so this
/// is how an application shows additional structures alongside it — other loaded
/// files, a reference pose, a docked ligand. Alpha travels per sphere, so a
/// whole set can be faded by the application without touching the main
/// molecule's opacity.
#[derive(Clone, Debug, Default)]
pub struct SphereSetState {
    pub sets: Vec<SphereSet>,
}

/// Draws every sphere in [`SphereSetState`] as an impostor, which stays cheap
/// even when the sets hold hundreds of thousands of atoms.
pub struct SphereSetRender;

impl SphereSetRender {
    pub fn new() -> Self {
        Self
    }
}

impl Default for SphereSetRender {
    fn default() -> Self {
        Self::new()
    }
}

impl AdditionalRender for SphereSetRender {
    fn gpu_pipeline(&self) -> GpuPipeline {
        GpuPipeline::SphereImpostor
    }

    fn update_scene(&self, scene: &mut Scene, frame_state: &RenderFrameState<'_>) {
        let Some(states) = frame_state.shared_states else {
            return;
        };

        with_state_by_type::<SphereSetState, ()>(states, |state| {
            let total: usize = state.sets.iter().map(|set| set.spheres.len()).sum();
            scene.sphere_impostors.reserve(total);

            for set in &state.sets {
                for sphere in &set.spheres {
                    self.add_sphere(
                        scene,
                        frame_state,
                        sphere.position,
                        sphere.radius,
                        sphere.color,
                    );
                }
            }
        });
    }
}
