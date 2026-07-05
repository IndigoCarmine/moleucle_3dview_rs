use crate::atom_radii::vdw_radius;
use crate::frame_state::RenderFrameState;
use crate::offscreen_renderer::RenderStyle;
use crate::render_state::get_state_clone_by_type;
use crate::scene_types::{Entity, Mesh, Scene, SphereImpostorInstance};
use lin_alg::f32::Quaternion;
use lin_alg::f32::Vec3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuPipeline {
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
        let sphere_mesh = Mesh::new_sphere_uv(1.0, mesh_resolution, mesh_resolution * 2);
        let mesh_idx = scene.meshes.len();
        scene.meshes.push(sphere_mesh);

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
        let cyl_mesh = Mesh::new_cylinder(1.0, 1.0, 10);
        let mesh_idx = scene.meshes.len();
        scene.meshes.push(cyl_mesh);

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
#[derive(Clone)]
pub struct SelectedAtomRenderState {
    pub selected_atoms: Vec<usize>,
    /// RGBA highlight color; the alpha component controls transparency.
    pub color: [f32; 4],
}
pub struct SelectedAtomRender {}

impl SelectedAtomRender {
    pub fn new() -> Self {
        Self {}
    }
}

impl AdditionalRender for SelectedAtomRender {
    fn update_scene(&self, scene: &mut Scene, frame: &RenderFrameState<'_>) {
        let Some(molecule) = frame.molecule else {
            return;
        };

        let Some(states) = frame.shared_states else {
            return;
        };

        // Use type-keyed state to lookup selected atoms. If not set, fall back to internal list.
        let source: SelectedAtomRenderState =
            get_state_clone_by_type::<SelectedAtomRenderState>(states).unwrap_or_else(|| {
                SelectedAtomRenderState {
                    selected_atoms: Vec::new(),
                    color: [1.0, 0.0, 0.0, 1.0], // Default red color
                }
            });

        let color = (
            source.color[0],
            source.color[1],
            source.color[2],
            source.color[3],
        );
        scene.entities.reserve(source.selected_atoms.len());
        for atom_idx in &source.selected_atoms {
            if let Some(atom) = molecule.atoms.get(*atom_idx) {
                self.add_sphere(
                    scene,
                    frame,
                    atom.position,
                    vdw_radius(&atom.element) * 0.4,
                    color,
                );
            }
        }
    }
}

impl SelectedAtomRenderState {
    pub fn new(selected_atoms: Vec<usize>, color: [f32; 4]) -> Self {
        Self {
            selected_atoms,
            color,
        }
    }

    pub fn set_color(&mut self, color: [f32; 4]) {
        self.color = color;
    }

    pub fn remove_atom(&mut self, atom_idx: usize) {
        if let Some(pos) = self.selected_atoms.iter().position(|&x| x == atom_idx) {
            self.selected_atoms.remove(pos);
        }
    }

    pub fn toggle_atom(&mut self, atom_idx: usize) {
        if let Some(pos) = self.selected_atoms.iter().position(|&x| x == atom_idx) {
            self.selected_atoms.remove(pos);
        } else {
            self.selected_atoms.push(atom_idx);
        }
    }
}
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

        let state = get_state_clone_by_type::<DebugRenderState>(states).unwrap_or_else(|| {
            DebugRenderState::new((
                Vec3::new(0.0, 0.0, 0.0),
                Vec3::new(1.0, 0.0, 0.0), // Default ray along +X axis
            ))
        });
        // Draw debug ray as a thin cylinder
        let (origin, direction) = state.ray;

        // Normalize direction
        let normalized_dir = direction.to_normalized();
        let ray_radius = 0.05; // Thin cylinder for visualization

        // Calculate midpoint of the ray
        let ray_end = origin + normalized_dir * state.ray_length;
        let midpoint = (origin + ray_end) * 0.5;

        // Create cylinder mesh
        let ray_mesh = Mesh::new_cylinder(1.0, 1.0, 8);
        let ray_idx = scene.meshes.len();
        scene.meshes.push(ray_mesh);

        // Quaternion to rotate from Y-axis (default cylinder orientation) to ray direction
        let up = Vec3::new(0.0, 1.0, 0.0);
        let orientation = Quaternion::from_unit_vecs(up, normalized_dir);

        // Create entity with proper scaling
        let mut ray_entity = Entity::new(ray_idx, midpoint, orientation, 1.0, state.ray_color, 0.1);

        // Apply scale_partial to set cylinder dimensions
        // X and Z are radii, Y is length
        ray_entity.scale_partial = Some(Vec3::new(ray_radius, state.ray_length, ray_radius));
        scene.entities.push(ray_entity);
    }
}
