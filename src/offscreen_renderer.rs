use crate::additional_render::{AdditionalRender, GpuPipeline};
use crate::frame_state::RenderFrameState;
use crate::render_state::SharedRenderStates;
use crate::scene_types::{Scene, SphereImpostorInstance};
use crate::viewer::ColorFn;
use crate::Molecule;
use egui::TextureId;
use egui_wgpu::wgpu;
use lin_alg::f32::Vec3;
use wgpu::util::DeviceExt;

mod gpu;
mod lod;
mod render_styles;

pub use lod::LodSettings;

use crate::atom_radii::{ball_stick_radius, default_ball_stick_bond_radius, vdw_radius};
use gpu::{create_gpu_resources, upload_instances, GpuResources};
use render_styles::circles::{
    fill_circle_instances, fill_sphere_instances, CircleInstance, MAX_IMPOSTOR_INSTANCES,
};
use render_styles::{style_for, StyleBuildContext};

const DEFAULT_MESH_RESOLUTION: usize = 3;
const DEFAULT_BOND_CYLINDER_SIDES: usize = 12;
const SAFE_MAX_VERTEX_BUFFER_BYTES: usize = 240 * 1024 * 1024;
const MAX_RENDER_VERTICES: usize = SAFE_MAX_VERTEX_BUFFER_BYTES / std::mem::size_of::<Vertex>();

/// Vertices in the lowest-quality UV sphere (lat=3, lon=6): 3 * 6 quads * 6.
const MIN_SPHERE_VERTICES_PER_ATOM: usize = 3 * 6 * 6;
/// Above this atom count the mesh path cannot fit even at lowest quality, so
/// the renderer auto-switches mesh styles to the sphere-impostor pipeline
/// instead of silently truncating atoms.
const MAX_MESH_ATOMS: usize = MAX_RENDER_VERTICES / MIN_SPHERE_VERTICES_PER_ATOM;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OffscreenRendererPreference {
    mesh_resolution: usize,
    lod_settings: LodSettings,
    render_style: RenderStyle,
    is_low_mode: bool,
}

impl Default for OffscreenRendererPreference {
    fn default() -> Self {
        Self {
            mesh_resolution: DEFAULT_MESH_RESOLUTION,
            lod_settings: LodSettings::default(),
            render_style: RenderStyle::BallStick,
            is_low_mode: false,
        }
    }
}

impl OffscreenRendererPreference {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_mesh_resolution(mesh_resolution: usize) -> Self {
        let mut preference = Self::default();
        preference.set_mesh_resolution(mesh_resolution);
        preference
    }

    pub fn mesh_resolution(&self) -> usize {
        self.mesh_resolution
    }

    pub fn lod_settings(&self) -> LodSettings {
        self.lod_settings
    }

    pub fn render_style(&self) -> RenderStyle {
        self.render_style
    }

    pub fn is_low_mode(&self) -> bool {
        self.is_low_mode
    }

    pub fn set_mesh_resolution(&mut self, mesh_resolution: usize) {
        self.mesh_resolution = mesh_resolution.max(3);
    }

    pub fn set_lod_settings(&mut self, lod_settings: LodSettings) {
        self.lod_settings = lod_settings;
    }

    pub fn set_render_style(&mut self, render_style: RenderStyle) {
        self.render_style = render_style;
    }

    pub fn set_is_low_mode(&mut self, low: bool) {
        self.is_low_mode = low;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderStyle {
    BallStick,
    BallOnly,
    Circles,
    Wireframe,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CircleQuadVertex {
    corner: [f32; 2],
}

/// Per-vertex data for the unit cylinder reused by every instanced bond.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BondMeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

/// Per-bond instance data. One cylinder mesh is drawn once per bond via GPU
/// instancing, so a molecule's bonds cost a single small mesh plus this packed
/// array instead of duplicating cylinder geometry per bond on the CPU.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct BondInstance {
    mid: [f32; 3],
    radius: f32,
    axis: [f32; 3],
    length: f32,
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    view_proj: [f32; 16],
    viewport: [f32; 2],
    focal: f32,
    _pad: f32,
    camera_right: [f32; 4],
    camera_up: [f32; 4],
    camera_forward: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GeometryCacheKey {
    molecule_ptr: usize,
    /// Changes when atom positions are updated in place (trajectory playback),
    /// so the cached geometry rebuilds even though the molecule pointer is the
    /// same object.
    generation: u64,
    render_style: RenderStyle,
    color_fn_ptr: usize,
    mesh_resolution: usize,
}

pub struct OffscreenRenderer {
    width: u32,
    height: u32,
    preference: OffscreenRendererPreference,
    color_texture: Option<wgpu::Texture>,
    depth_texture: Option<wgpu::Texture>,
    texture_id: Option<TextureId>,
    gpu: Option<GpuResources>,
    sphere_mesh: RenderMesh,
    cylinder_mesh: RenderMesh,
    geometry_cache_key: Option<GeometryCacheKey>,
    lod: lod::LodManager,
    additional_renders: Vec<Box<dyn AdditionalRender>>,
    /// Reusable CPU scratch buffers for impostor/bond instances, kept across
    /// frames so trajectory playback refills them without reallocating.
    scratch_circle_instances: Vec<CircleInstance>,
    scratch_bond_instances: Vec<BondInstance>,
}

impl OffscreenRenderer {
    pub fn new() -> Self {
        Self::new_with_preference(OffscreenRendererPreference::default())
    }

    pub fn new_with_mesh_resolution(mesh_resolution: usize) -> Self {
        Self::new_with_preference(OffscreenRendererPreference::with_mesh_resolution(
            mesh_resolution,
        ))
    }

    pub fn new_with_mesh_resolution_and_lod(
        mesh_resolution: usize,
        lod_settings: LodSettings,
    ) -> Self {
        let mut preference = OffscreenRendererPreference::with_mesh_resolution(mesh_resolution);
        preference.set_lod_settings(lod_settings);
        Self::new_with_preference(preference)
    }

    pub fn new_with_preference(preference: OffscreenRendererPreference) -> Self {
        let mesh_resolution = preference.mesh_resolution().max(3);
        let sphere_lat = mesh_resolution;
        let sphere_lon = mesh_resolution * 2;
        let cylinder_sides = DEFAULT_BOND_CYLINDER_SIDES;

        Self {
            width: 0,
            height: 0,
            lod: lod::LodManager::new(preference.lod_settings()),
            preference,
            color_texture: None,
            depth_texture: None,
            texture_id: None,
            gpu: None,
            sphere_mesh: RenderMesh::new_sphere_uv(1.0, sphere_lat, sphere_lon),
            cylinder_mesh: RenderMesh::new_cylinder_open_ended(1.0, 1.0, cylinder_sides),
            geometry_cache_key: None,
            additional_renders: Vec::new(),
            scratch_circle_instances: Vec::new(),
            scratch_bond_instances: Vec::new(),
        }
    }

    pub fn set_mesh_resolution(&mut self, mesh_resolution: usize) {
        let mesh_resolution = mesh_resolution.max(3);
        if self.preference.mesh_resolution() == mesh_resolution {
            return;
        }

        self.preference.set_mesh_resolution(mesh_resolution);
        let sphere_lat = mesh_resolution;
        let sphere_lon = mesh_resolution * 2;
        let cylinder_sides = DEFAULT_BOND_CYLINDER_SIDES;
        self.sphere_mesh = RenderMesh::new_sphere_uv(1.0, sphere_lat, sphere_lon);
        self.cylinder_mesh = RenderMesh::new_cylinder_open_ended(1.0, 1.0, cylinder_sides);
        self.geometry_cache_key = None;
    }

    pub fn mesh_resolution(&self) -> usize {
        self.preference.mesh_resolution()
    }

    pub fn lod_settings(&self) -> LodSettings {
        self.preference.lod_settings()
    }

    pub fn set_lod_settings(&mut self, lod_settings: LodSettings) {
        if self.preference.lod_settings() == lod_settings {
            return;
        }

        self.preference.set_lod_settings(lod_settings);
        self.lod.update_settings(lod_settings);
        self.geometry_cache_key = None;
    }

    pub fn render_style(&self) -> RenderStyle {
        self.preference.render_style()
    }

    pub fn is_low_mode(&self) -> bool {
        self.preference.is_low_mode()
    }

    pub fn set_render_style(&mut self, render_style: RenderStyle) {
        if self.preference.render_style() != render_style {
            self.preference.set_render_style(render_style);
            self.geometry_cache_key = None;
        }
    }

    pub fn set_is_low_mode(&mut self, low: bool) {
        self.preference.set_is_low_mode(low);
        self.geometry_cache_key = None;
    }

    pub fn preference(&self) -> OffscreenRendererPreference {
        self.preference
    }

    pub fn set_preference(&mut self, preference: OffscreenRendererPreference) {
        if self.preference == preference {
            return;
        }

        let mesh_resolution_changed =
            self.preference.mesh_resolution() != preference.mesh_resolution();
        let lod_changed = self.preference.lod_settings() != preference.lod_settings();
        self.preference = preference;

        if mesh_resolution_changed {
            let mesh_resolution = self.preference.mesh_resolution();
            let sphere_lat = mesh_resolution;
            let sphere_lon = mesh_resolution * 2;
            let cylinder_sides = DEFAULT_BOND_CYLINDER_SIDES;
            self.sphere_mesh = RenderMesh::new_sphere_uv(1.0, sphere_lat, sphere_lon);
            self.cylinder_mesh = RenderMesh::new_cylinder_open_ended(1.0, 1.0, cylinder_sides);
        }

        if lod_changed {
            self.lod.update_settings(self.preference.lod_settings());
        }

        self.geometry_cache_key = None;
    }

    pub fn texture_id(&self) -> Option<TextureId> {
        self.texture_id
    }

    pub fn ensure_resources(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        width: u32,
        height: u32,
    ) -> Result<(), String> {
        if width == 0 || height == 0 {
            return Err("Offscreen size must be non-zero".to_string());
        }

        if self.gpu.is_none() {
            self.gpu = Some(create_gpu_resources(&render_state.device));
        }

        let needs_rebuild =
            self.color_texture.is_none() || self.width != width || self.height != height;
        if needs_rebuild {
            self.width = width;
            self.height = height;
            self.rebuild_targets(render_state);
        }

        Ok(())
    }

    pub fn add_additional_render(&mut self, render: Box<dyn AdditionalRender>) {
        self.additional_renders.push(render);
    }

    fn additional_pipeline_for<'a>(
        gpu: &'a GpuResources,
        pipeline: GpuPipeline,
    ) -> &'a wgpu::RenderPipeline {
        match pipeline {
            GpuPipeline::Triangles => &gpu.additional_pipeline,
            GpuPipeline::Wireframe => &gpu.wire_pipeline,
            GpuPipeline::SphereImpostor => &gpu.circles_pipeline,
        }
    }

    pub fn render_frame_with_state(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        frame: &RenderFrameState<'_>,
    ) -> Result<(), String> {
        self.apply_pending_lod_resolution();

        let color_texture = self
            .color_texture
            .as_ref()
            .ok_or_else(|| "Offscreen color texture is not initialized".to_string())?;
        let depth_texture = self
            .depth_texture
            .as_ref()
            .ok_or_else(|| "Offscreen depth texture is not initialized".to_string())?;

        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let render_style = self.preference.render_style();
        let atom_count = frame.molecule.map(|mol| mol.atoms.len()).unwrap_or(0);
        // Mesh styles duplicate sphere geometry per atom and would overflow the
        // vertex buffer past MAX_MESH_ATOMS, silently dropping atoms. Fall back
        // to the instanced impostor pipeline so every atom is still drawn.
        let mesh_overflow = matches!(render_style, RenderStyle::BallStick | RenderStyle::BallOnly)
            && atom_count > MAX_MESH_ATOMS;
        let use_impostors = matches!(render_style, RenderStyle::Circles) || mesh_overflow;

        let style = if use_impostors {
            None
        } else {
            Some(style_for(render_style))
        };

        let cache_key = self.build_geometry_cache_key(frame.molecule, frame.color_fn);
        let rebuilt_vertices = if !use_impostors && self.geometry_cache_key != Some(cache_key) {
            if let Some(active_style) = style {
                let ctx = StyleBuildContext {
                    preference: self.preference,
                    sphere_mesh: &self.sphere_mesh,
                    cylinder_mesh: &self.cylinder_mesh,
                    molecule_opacity: frame.molecule_opacity,
                };
                Some(active_style.build_vertices(&ctx, frame.molecule, frame.color_fn))
            } else {
                None
            }
        } else {
            None
        };

        // Impostor instances don't depend on the camera, so rebuild them only
        // when the molecule/style/color/positions change — not every frame. The
        // generation in cache_key catches in-place trajectory updates. Refill
        // the reusable scratch Vecs (moved out to avoid reallocation) so an
        // idle 500k-atom view, and equal-size trajectory frames, don't churn.
        let cache_miss = self.geometry_cache_key != Some(cache_key);
        let rebuild_impostors = use_impostors && cache_miss;

        // When BallStick falls back to impostors, render its bonds via the
        // instanced cylinder pipeline (one instance per bond) so connectivity
        // still shows at a scale where the per-bond CPU mesh would overflow.
        let bonds_as_instances = mesh_overflow && matches!(render_style, RenderStyle::BallStick);
        let rebuild_bonds = bonds_as_instances && cache_miss;

        let mut circle_scratch = std::mem::take(&mut self.scratch_circle_instances);
        let mut bond_scratch = std::mem::take(&mut self.scratch_bond_instances);
        if rebuild_impostors {
            Self::fill_impostor_instances(&mut circle_scratch, render_style, frame);
        }
        if rebuild_bonds {
            Self::fill_bond_instances(&mut bond_scratch, frame);
        }

        let additional_batches: Vec<(GpuPipeline, Vec<Vertex>, Vec<SphereImpostorInstance>)> = self
            .additional_renders
            .iter()
            .map(|additional| {
                let mut additional_scene = Scene {
                    meshes: Vec::new(),
                    entities: Vec::new(),
                    sphere_impostors: Vec::new(),
                };
                additional.update_scene(&mut additional_scene, frame);
                (
                    additional.gpu_pipeline(),
                    self.build_additional_scene_vertices(&additional_scene),
                    additional_scene.sphere_impostors,
                )
            })
            .collect();

        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| "Offscreen GPU resources are not initialized".to_string())?;

        if let Some(mut vertices) = rebuilt_vertices {
            let primitive_stride = style
                .map(|active_style| active_style.primitive_stride())
                .unwrap_or(3);

            if vertices.len() > MAX_RENDER_VERTICES {
                let capped = MAX_RENDER_VERTICES - (MAX_RENDER_VERTICES % primitive_stride);
                vertices.truncate(capped);
            }

            gpu.vertex_count = upload_instances(
                &render_state.device,
                &render_state.queue,
                &mut gpu.vertex_buffer,
                &mut gpu.vertex_capacity,
                "offscreen-vertex-buffer",
                &vertices,
            );
            self.geometry_cache_key = Some(cache_key);
        }

        if rebuild_impostors {
            gpu.circles_instance_count = upload_instances(
                &render_state.device,
                &render_state.queue,
                &mut gpu.circles_instance_buffer,
                &mut gpu.circles_instance_capacity,
                "offscreen-circles-instance-buffer",
                &circle_scratch,
            );
            self.geometry_cache_key = Some(cache_key);
        }

        if rebuild_bonds {
            gpu.bond_instance_count = upload_instances(
                &render_state.device,
                &render_state.queue,
                &mut gpu.bond_instance_buffer,
                &mut gpu.bond_instance_capacity,
                "offscreen-bond-instance-buffer",
                &bond_scratch,
            );
            self.geometry_cache_key = Some(cache_key);
        }

        let focal = 1.0 / (frame.fov_y * 0.5).tan();
        let uniforms = Uniforms {
            view_proj: frame.view_proj,
            viewport: [self.width as f32, self.height as f32],
            focal,
            _pad: 0.0,
            camera_right: [
                frame.camera_right.x,
                frame.camera_right.y,
                frame.camera_right.z,
                0.0,
            ],
            camera_up: [frame.camera_up.x, frame.camera_up.y, frame.camera_up.z, 0.0],
            camera_forward: [
                frame.camera_forward.x,
                frame.camera_forward.y,
                frame.camera_forward.z,
                0.0,
            ],
        };
        render_state
            .queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut encoder =
            render_state
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("offscreen-render-encoder"),
                });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("offscreen-pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.08,
                            g: 0.10,
                            b: 0.14,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            let pipeline = if use_impostors {
                &gpu.circles_pipeline
            } else {
                match self.preference.render_style() {
                    RenderStyle::BallStick if self.preference.is_low_mode() => &gpu.wire_pipeline,
                    RenderStyle::BallStick => &gpu.pipeline,
                    RenderStyle::BallOnly => &gpu.pipeline,
                    RenderStyle::Circles => &gpu.circles_pipeline,
                    RenderStyle::Wireframe => &gpu.wire_pipeline,
                }
            };

            pass.set_pipeline(pipeline);

            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            if use_impostors {
                if let Some(instance_buffer) = &gpu.circles_instance_buffer {
                    pass.set_vertex_buffer(0, gpu.circles_quad_buffer.slice(..));
                    pass.set_vertex_buffer(1, instance_buffer.slice(..));
                    pass.draw(0..6, 0..gpu.circles_instance_count);
                }

                // Instanced bonds for the BallStick large-molecule fallback.
                if bonds_as_instances {
                    if let Some(bond_buffer) = &gpu.bond_instance_buffer {
                        if gpu.bond_instance_count > 0 {
                            pass.set_pipeline(&gpu.bond_pipeline);
                            pass.set_vertex_buffer(0, gpu.bond_mesh_buffer.slice(..));
                            pass.set_vertex_buffer(1, bond_buffer.slice(..));
                            pass.draw(
                                0..gpu.bond_mesh_vertex_count,
                                0..gpu.bond_instance_count,
                            );
                        }
                    }
                }
            } else if let Some(vertex_buffer) = &gpu.vertex_buffer {
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.draw(0..gpu.vertex_count, 0..1);
            }

            for (pipeline_kind, additional_vertices, additional_sphere_impostors) in
                additional_batches.into_iter()
            {
                if !additional_vertices.is_empty() && pipeline_kind != GpuPipeline::SphereImpostor {
                    let additional_vertex_buffer = render_state.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("offscreen-additional-vertex-buffer"),
                            contents: bytemuck::cast_slice(&additional_vertices),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    );

                    let pipeline = Self::additional_pipeline_for(gpu, pipeline_kind);
                    pass.set_pipeline(pipeline);
                    pass.set_vertex_buffer(0, additional_vertex_buffer.slice(..));
                    pass.draw(0..additional_vertices.len() as u32, 0..1);
                }

                if !additional_sphere_impostors.is_empty() {
                    let sphere_instance_buffer = render_state.device.create_buffer_init(
                        &wgpu::util::BufferInitDescriptor {
                            label: Some("offscreen-additional-sphere-instance-buffer"),
                            contents: bytemuck::cast_slice(&additional_sphere_impostors),
                            usage: wgpu::BufferUsages::VERTEX,
                        },
                    );

                    let pipeline = Self::additional_pipeline_for(gpu, GpuPipeline::SphereImpostor);
                    pass.set_pipeline(pipeline);
                    pass.set_vertex_buffer(0, gpu.circles_quad_buffer.slice(..));
                    pass.set_vertex_buffer(1, sphere_instance_buffer.slice(..));
                    pass.draw(0..6, 0..additional_sphere_impostors.len() as u32);
                }
            }
        }

        render_state.queue.submit(std::iter::once(encoder.finish()));

        // Return the scratch buffers (with their grown capacity) so the next
        // frame refills them without reallocating.
        self.scratch_circle_instances = circle_scratch;
        self.scratch_bond_instances = bond_scratch;
        Ok(())
    }

    pub fn render_frame(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        molecule: Option<&Molecule>,
        view_proj: [f32; 16],
        color_fn: ColorFn,
        additionalstate: Option<SharedRenderStates>,
    ) -> Result<(), String> {
        let frame = RenderFrameState::new(
            molecule,
            view_proj,
            None,
            std::f32::consts::FRAC_PI_4,
            lin_alg::f32::Vec3::new(1.0, 0.0, 0.0),
            lin_alg::f32::Vec3::new(0.0, 1.0, 0.0),
            lin_alg::f32::Vec3::new(0.0, 0.0, 1.0),
            color_fn,
            additionalstate.as_ref(),
            self.preference.render_style(),
            self.preference.mesh_resolution(),
            self.preference.is_low_mode(),
            1.0,
        );
        self.render_frame_with_state(render_state, &frame)
    }

    pub fn submit_lod_distance(&self, distance: f32) {
        self.lod.submit_distance(distance);
    }

    fn apply_pending_lod_resolution(&mut self) {
        let Some(target_resolution) = self.lod.poll_resolution() else {
            return;
        };

        self.set_mesh_resolution(target_resolution.max(3));
    }

    pub fn free_egui_texture(&mut self, render_state: &egui_wgpu::RenderState) {
        if let Some(id) = self.texture_id.take() {
            let mut renderer = render_state.renderer.write();
            renderer.free_texture(&id);
        }
    }

    fn rebuild_targets(&mut self, render_state: &egui_wgpu::RenderState) {
        let device = &render_state.device;

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen-color"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen-depth"),
            size: wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth24Plus,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });

        {
            let mut renderer = render_state.renderer.write();
            if let Some(id) = self.texture_id {
                renderer.update_egui_texture_from_wgpu_texture(
                    device,
                    &color_view,
                    wgpu::FilterMode::Linear,
                    id,
                );
            } else {
                self.texture_id = Some(renderer.register_native_texture(
                    device,
                    &color_view,
                    wgpu::FilterMode::Linear,
                ));
            }
        }

        self.color_texture = Some(color_texture);
        self.depth_texture = Some(depth_texture);
    }

    /// Fill `out` with one cylinder instance per bond for the instanced bond
    /// pipeline. Used by the large-molecule BallStick fallback so bonds still
    /// render at a scale where the per-bond CPU mesh path would overflow the
    /// vertex buffer. `out`'s capacity is reused across frames.
    fn fill_bond_instances(out: &mut Vec<BondInstance>, frame: &RenderFrameState<'_>) {
        out.clear();
        let Some(mol) = frame.molecule else {
            return;
        };

        let radius = default_ball_stick_bond_radius();
        let color = [0.55, 0.55, 0.55, 1.0];
        out.reserve(mol.bonds.len().min(MAX_IMPOSTOR_INSTANCES));

        for bond in &mol.bonds {
            if out.len() >= MAX_IMPOSTOR_INSTANCES {
                break;
            }
            let a = mol.atoms[bond.atom_a].position;
            let b = mol.atoms[bond.atom_b].position;
            let diff = b - a;
            let len = diff.magnitude();
            if len < 1e-4 {
                continue;
            }
            let axis = diff / len;
            let mid = (a + b) * 0.5;
            out.push(BondInstance {
                mid: [mid.x, mid.y, mid.z],
                radius,
                axis: [axis.x, axis.y, axis.z],
                length: len,
                color,
            });
        }
    }

    /// Fill `out` with sphere-impostor instances for the active style, choosing
    /// a radius that matches the mesh each style would otherwise draw. `out`'s
    /// capacity is reused across frames.
    fn fill_impostor_instances(
        out: &mut Vec<CircleInstance>,
        render_style: RenderStyle,
        frame: &RenderFrameState<'_>,
    ) {
        let opacity = frame.molecule_opacity;
        match render_style {
            RenderStyle::Circles => {
                fill_circle_instances(out, frame.molecule, frame.color_fn, opacity)
            }
            RenderStyle::BallOnly => {
                fill_sphere_instances(out, frame.molecule, frame.color_fn, opacity, |atom| {
                    vdw_radius(&atom.element)
                })
            }
            // BallStick falls back here only when the mesh path would overflow;
            // bonds are dropped at that scale, but every atom is still shown.
            RenderStyle::BallStick => {
                fill_sphere_instances(out, frame.molecule, frame.color_fn, opacity, |atom| {
                    ball_stick_radius(&atom.element, false)
                })
            }
            RenderStyle::Wireframe => out.clear(),
        }
    }

    fn build_geometry_cache_key(
        &self,
        molecule: Option<&Molecule>,
        color_fn: ColorFn,
    ) -> GeometryCacheKey {
        GeometryCacheKey {
            molecule_ptr: molecule
                .map(|mol| mol as *const Molecule as usize)
                .unwrap_or(0),
            generation: molecule.map(|mol| mol.generation()).unwrap_or(0),
            render_style: self.preference.render_style(),
            color_fn_ptr: color_fn as usize,
            mesh_resolution: self.preference.mesh_resolution(),
        }
    }

    fn build_additional_scene_vertices(&self, scene: &Scene) -> Vec<Vertex> {
        let max_vertices = MAX_RENDER_VERTICES;
        let mut vertices = Vec::new();

        for entity in &scene.entities {
            let Some(mesh) = scene.meshes.get(entity.mesh) else {
                continue;
            };

            let scale =
                entity
                    .scale_partial
                    .unwrap_or(Vec3::new(entity.scale, entity.scale, entity.scale));
            // Fold the entity's opacity into the color's alpha channel so both
            // per-color alpha and the entity-wide opacity affect blending.
            let color = [
                entity.color.0,
                entity.color.1,
                entity.color.2,
                entity.color.3 * entity.opacity,
            ];

            for tri in mesh.indices.chunks_exact(3) {
                if vertices.len().saturating_add(3) > max_vertices {
                    return vertices;
                }

                for &idx in tri {
                    let Some(src) = mesh.vertices.get(idx) else {
                        return vertices;
                    };

                    let p = Vec3::new(src.position[0], src.position[1], src.position[2]);
                    let p_scaled = Vec3::new(p.x * scale.x, p.y * scale.y, p.z * scale.z);
                    let p_world = entity.orientation.rotate_vec(p_scaled) + entity.position;

                    let n = src.normal;
                    let n_world = entity.orientation.rotate_vec(n).to_normalized();

                    vertices.push(Vertex {
                        position: [p_world.x, p_world.y, p_world.z],
                        normal: [n_world.x, n_world.y, n_world.z],
                        color,
                    });
                }
            }
        }

        vertices
    }
}

impl Default for OffscreenRenderer {
    fn default() -> Self {
        Self::new()
    }
}

fn bond_line_offsets(order: usize) -> Vec<f32> {
    const DEFAULT_BOND_SPACING: f32 = 0.01;
    match order {
        0 | 1 => vec![0.0],
        2 => vec![-DEFAULT_BOND_SPACING, DEFAULT_BOND_SPACING],
        3 => vec![-DEFAULT_BOND_SPACING, 0.0, DEFAULT_BOND_SPACING],
        n => {
            let spacing = DEFAULT_BOND_SPACING;
            let half = (n as f32 - 1.0) * 0.5;
            (0..n).map(|i| (i as f32 - half) * spacing).collect()
        }
    }
}

fn append_mesh_triangles(
    out: &mut Vec<Vertex>,
    mesh: &RenderMesh,
    position: lin_alg::f32::Vec3,
    orientation: lin_alg::f32::Quaternion,
    scale: lin_alg::f32::Vec3,
    color: [f32; 4],
    max_vertices: usize,
) -> bool {
    let inv_scale = lin_alg::f32::Vec3::new(
        if scale.x.abs() > 1e-6 { 1.0 / scale.x } else { 0.0 },
        if scale.y.abs() > 1e-6 { 1.0 / scale.y } else { 0.0 },
        if scale.z.abs() > 1e-6 { 1.0 / scale.z } else { 0.0 },
    );

    for tri in mesh.indices.chunks_exact(3) {
        if out.len().saturating_add(3) > max_vertices {
            return false;
        }

        for &idx in tri {
            let Some(src) = mesh.vertices.get(idx) else {
                return false;
            };

            let p = lin_alg::f32::Vec3::new(src.position[0], src.position[1], src.position[2]);
            let p_scaled = lin_alg::f32::Vec3::new(p.x * scale.x, p.y * scale.y, p.z * scale.z);
            let p_world = orientation.rotate_vec(p_scaled) + position;

            let n = lin_alg::f32::Vec3::new(src.normal[0], src.normal[1], src.normal[2]);
            let n_scaled = lin_alg::f32::Vec3::new(
                n.x * inv_scale.x,
                n.y * inv_scale.y,
                n.z * inv_scale.z,
            );
            let n_world = orientation.rotate_vec(n_scaled).to_normalized();

            out.push(Vertex {
                position: [p_world.x, p_world.y, p_world.z],
                normal: [n_world.x, n_world.y, n_world.z],
                color,
            });
        }
    }

    true
}

#[derive(Clone, Copy)]
struct RenderVertex {
    position: [f32; 3],
    normal: [f32; 3],
}

struct RenderMesh {
    vertices: Vec<RenderVertex>,
    indices: Vec<usize>,
}

impl RenderMesh {
    fn new_sphere_uv(radius: f32, lat_segments: usize, lon_segments: usize) -> Self {
        let vertex_count = (lat_segments + 1) * (lon_segments + 1);
        let index_count = lat_segments * lon_segments * 6;

        let mut vertices = Vec::with_capacity(vertex_count);
        let mut indices = Vec::with_capacity(index_count);

        for lat in 0..=lat_segments {
            let v = lat as f32 / lat_segments as f32;
            let theta = v * std::f32::consts::PI;
            let sin_t = theta.sin();
            let cos_t = theta.cos();

            for lon in 0..=lon_segments {
                let u = lon as f32 / lon_segments as f32;
                let phi = u * std::f32::consts::TAU;
                let sin_p = phi.sin();
                let cos_p = phi.cos();

                let x = radius * sin_t * cos_p;
                let y = radius * cos_t;
                let z = radius * sin_t * sin_p;
                let n = lin_alg::f32::Vec3::new(x, y, z).to_normalized();

                vertices.push(RenderVertex {
                    position: [x, y, z],
                    normal: [n.x, n.y, n.z],
                });
            }
        }

        let row = lon_segments + 1;
        for lat in 0..lat_segments {
            for lon in 0..lon_segments {
                let i0 = lat * row + lon;
                let i1 = i0 + 1;
                let i2 = i0 + row;
                let i3 = i2 + 1;

                indices.push(i0);
                indices.push(i2);
                indices.push(i1);
                indices.push(i1);
                indices.push(i2);
                indices.push(i3);
            }
        }

        Self { vertices, indices }
    }

    fn new_cylinder_open_ended(len: f32, radius: f32, sides: usize) -> Self {
        let vertex_capacity = sides * 2;
        let index_capacity = sides * 6;

        let mut vertices = Vec::with_capacity(vertex_capacity);
        let mut indices = Vec::with_capacity(index_capacity);
        let half = len * 0.5;

        for i in 0..sides {
            let t = i as f32 / sides as f32;
            let angle = t * std::f32::consts::TAU;
            let x = radius * angle.cos();
            let z = radius * angle.sin();
            let n = lin_alg::f32::Vec3::new(x, 0.0, z).to_normalized();

            vertices.push(RenderVertex {
                position: [x, half, z],
                normal: [n.x, n.y, n.z],
            });
            vertices.push(RenderVertex {
                position: [x, -half, z],
                normal: [n.x, n.y, n.z],
            });
        }

        for i in 0..sides {
            let next = (i + 1) % sides;
            let top0 = i * 2;
            let bot0 = top0 + 1;
            let top1 = next * 2;
            let bot1 = top1 + 1;

            indices.push(top0);
            indices.push(bot0);
            indices.push(top1);
            indices.push(top1);
            indices.push(bot0);
            indices.push(bot1);
        }

        Self { vertices, indices }
    }
}

fn append_line(
    out: &mut Vec<Vertex>,
    a: lin_alg::f32::Vec3,
    b: lin_alg::f32::Vec3,
    color: [f32; 4],
    max_vertices: usize,
) -> bool {
    if out.len().saturating_add(2) > max_vertices {
        return false;
    }

    let normal = [0.0, 1.0, 0.0];
    out.push(Vertex {
        position: [a.x, a.y, a.z],
        normal,
        color,
    });
    out.push(Vertex {
        position: [b.x, b.y, b.z],
        normal,
        color,
    });

    true
}
