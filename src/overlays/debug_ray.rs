//! Debug overlay drawing a ray as a thin cylinder.

use crate::frame_state::RenderFrameState;
use crate::render_state::with_state_by_type;
use crate::scene_types::{Entity, Scene};
use crate::AdditionalRender;
use lin_alg::f32::{Quaternion, Vec3};

#[derive(Clone)]
pub struct DebugRenderState {
    pub ray: (Vec3, Vec3),
    pub ray_length: f32,
    pub ray_color: (f32, f32, f32, f32),
}

impl DebugRenderState {
    pub fn new(ray: (Vec3, Vec3)) -> Self {
        Self {
            ray,
            ray_length: 100.0,
            ray_color: (0.0, 1.0, 0.0, 1.0), // Default green color
        }
    }

    /// Create a debug renderer with custom ray length
    pub fn with_length(ray: (Vec3, Vec3), length: f32) -> Self {
        Self {
            ray,
            ray_length: length,
            ray_color: (0.0, 1.0, 0.0, 1.0),
        }
    }

    /// Create a debug renderer with custom color
    pub fn with_color(ray: (Vec3, Vec3), color: (f32, f32, f32, f32)) -> Self {
        Self {
            ray,
            ray_length: 100.0,
            ray_color: color,
        }
    }

    pub fn set_ray_length(&mut self, length: f32) {
        self.ray_length = length;
    }

    pub fn set_ray_color(&mut self, color: (f32, f32, f32, f32)) {
        self.ray_color = color;
    }
}

pub struct DebugRender {}

impl AdditionalRender for DebugRender {
    fn update_scene(&self, scene: &mut Scene, frame: &RenderFrameState<'_>) {
        let Some(states) = frame.shared_states else {
            return;
        };

        // `DebugRenderState` is all `Copy` scalars, so take a snapshot and let
        // the lock go before touching the scene.
        let state = with_state_by_type::<DebugRenderState, _>(states, |state| {
            (state.ray, state.ray_length, state.ray_color)
        });
        let Some((ray, ray_length, ray_color)) = state else {
            return;
        };

        // Draw debug ray as a thin cylinder
        let (origin, direction) = ray;

        // Normalize direction
        let normalized_dir = direction.to_normalized();
        let ray_radius = 0.05; // Thin cylinder for visualization

        // Calculate midpoint of the ray
        let ray_end = origin + normalized_dir * ray_length;
        let midpoint = (origin + ray_end) * 0.5;

        let ray_idx = scene.unit_cylinder_mesh(8);

        // Quaternion to rotate from Y-axis (default cylinder orientation) to ray direction
        let up = Vec3::new(0.0, 1.0, 0.0);
        let orientation = Quaternion::from_unit_vecs(up, normalized_dir);

        // Create entity with proper scaling
        let mut ray_entity = Entity::new(ray_idx, midpoint, orientation, 1.0, ray_color, 0.1);

        // Apply scale_partial to set cylinder dimensions
        // X and Z are radii, Y is length
        ray_entity.scale_partial = Some(Vec3::new(ray_radius, ray_length, ray_radius));
        scene.entities.push(ray_entity);
    }
}
