use crate::atom_radii::vdw_radius;
use crate::frame_state::RenderFrameState;
use crate::offscreen_renderer::RenderStyle;
use crate::scene_types::{Entity, Scene, SphereImpostorInstance};
use lin_alg::f32::Quaternion;
use lin_alg::f32::Vec3;

/// Radial segments in the cylinder [`AdditionalRender::add_cylinder`] draws.
/// Overlay cylinders are thin sticks and box edges, so a low count reads the
/// same and keeps the shared mesh small.
const CYLINDER_SIDES: usize = 10;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuPipeline {
    #[default]
    Triangles,
    Wireframe,
    SphereImpostor,
}

// for adding rendering works to MoleculeViewer.
pub trait AdditionalRender: Send {
    /// Update the given `scene` using `molecule` and the shared `states` map.
    fn update_scene(&self, scene: &mut Scene, frame: &RenderFrameState<'_>);

    /// Select which GPU pipeline should be used for this render contribution.
    fn gpu_pipeline(&self) -> GpuPipeline {
        GpuPipeline::Triangles
    }

    /// A value that changes whenever this overlay would draw something
    /// different, or `None` if it cannot say.
    ///
    /// `None` — the default — means "rebuild me every frame", which is always
    /// correct. Returning `Some` lets the renderer skip `update_scene`, the
    /// vertex build and the GPU upload entirely while the value holds still,
    /// and redraw the retained buffers instead. For an overlay whose cost
    /// scales with the structure that is the difference between a few
    /// microseconds and several milliseconds of CPU time *per frame*, spent
    /// re-deriving geometry nobody changed.
    ///
    /// Almost every implementation should be
    /// [`RenderFrameState::overlay_revision`] over its own state type, which
    /// folds in the molecule and the render settings as well. Getting this
    /// wrong makes the overlay silently stop updating, so a hand-rolled value
    /// needs to account for every input `update_scene` reads.
    fn revision(&self, _frame: &RenderFrameState<'_>) -> Option<u64> {
        None
    }

    fn add_sphere(
        &self,
        scene: &mut Scene,
        frame: &RenderFrameState<'_>,
        position: Vec3,
        radius: f32,
        color: (f32, f32, f32, f32),
    ) {
        if matches!(self.gpu_pipeline(), GpuPipeline::SphereImpostor)
            || matches!(frame.render_style, RenderStyle::Circles)
        {
            scene.sphere_impostors.push(SphereImpostorInstance {
                center: [position.x, position.y, position.z],
                radius,
                color: [color.0, color.1, color.2, color.3],
            });
            return;
        }

        let mesh_resolution = if frame.is_low_mode {
            frame.mesh_resolution.saturating_div(2).max(3)
        } else {
            frame.mesh_resolution.max(3)
        };
        // One unit sphere per resolution per scene, not one per call: overlays
        // that highlight many atoms call this once per atom per frame.
        let mesh_idx = scene.unit_sphere_mesh(mesh_resolution, mesh_resolution * 2);

        let entity = Entity::new(
            mesh_idx,
            position,
            Quaternion::new_identity(),
            radius,
            color,
            0.2,
        );
        scene.entities.push(entity);
    }

    fn add_sphere_sameas_carbon(
        &self,
        scene: &mut Scene,
        frame: &RenderFrameState<'_>,
        position: Vec3,
        relative_radius: f32,
        color: (f32, f32, f32, f32),
    ) {
        let radius = vdw_radius("C") * relative_radius; // Carbon van der Waals radius for demo
        self.add_sphere(scene, frame, position, radius, color);
    }
    fn add_cylinder(
        &self,
        scene: &mut Scene,
        start: Vec3,
        end: Vec3,
        radius: f32,
        color: (f32, f32, f32, f32),
    ) {
        // Shared across every cylinder in the scene; unlike `add_sphere` this
        // path has no impostor short-circuit, so it is the hotter of the two.
        let mesh_idx = scene.unit_cylinder_mesh(CYLINDER_SIDES);

        let mid_point = (start + end) * 0.5;
        let direction = end - start;
        let length = direction.magnitude();
        let orientation =
            Quaternion::from_unit_vecs(Vec3::new(0.0, 1.0, 0.0), direction.to_normalized());

        let mut entity = Entity::new(mesh_idx, mid_point, orientation, 1.0, color, 0.2);
        entity.scale_partial = Some(Vec3::new(radius, length, radius));
        scene.entities.push(entity);
    }
}
