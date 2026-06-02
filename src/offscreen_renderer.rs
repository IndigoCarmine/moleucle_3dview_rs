use crate::additional_render::{AdditionalRender, GpuPipeline};
use crate::atom_radii::{ball_stick_radius, default_ball_stick_bond_radius};
use crate::frame_state::RenderFrameState;
use crate::render_state::SharedRenderStates;
use crate::scene_types::{Scene, SphereImpostorInstance};
use crate::viewer::ColorFn;
use crate::Molecule;
use egui::TextureId;
use egui_wgpu::wgpu;
use lin_alg::f32::{Quaternion, Vec3};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use wgpu::util::DeviceExt;

mod render_styles;
use self::render_styles::circles::{build_circle_instances, CircleInstance};
use self::render_styles::style_for;

const DEFAULT_MESH_RESOLUTION: usize = 3;
const DEFAULT_BOND_CYLINDER_SIDES: usize = 12;
const SAFE_MAX_VERTEX_BUFFER_BYTES: usize = 240 * 1024 * 1024;
const MAX_RENDER_VERTICES: usize = SAFE_MAX_VERTEX_BUFFER_BYTES / std::mem::size_of::<Vertex>();

#[derive(Debug)]
struct VertexBufferBatch {
    buffer: wgpu::Buffer,
    count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LodSettings {
    pub enabled: bool,
    pub distance_check_fps: f32,
    pub high_detail_max_distance: f32,
    pub medium_detail_max_distance: f32,
    pub high_detail_mesh_resolution: usize,
    pub medium_detail_mesh_resolution: usize,
    pub low_detail_mesh_resolution: usize,
}

impl Default for LodSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            distance_check_fps: 12.0,
            high_detail_max_distance: 4.0,
            medium_detail_max_distance: 10.0,
            high_detail_mesh_resolution: 14,
            medium_detail_mesh_resolution: 8,
            low_detail_mesh_resolution: 4,
        }
    }
}

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

    pub fn set_mesh_resolution(&mut self, mesh_resolution: usize) {
        self.mesh_resolution = mesh_resolution.max(3);
    }

    pub fn set_lod_settings(&mut self, lod_settings: LodSettings) {
        self.lod_settings = lod_settings;
    }

    pub fn set_render_style(&mut self, render_style: RenderStyle) {
        self.render_style = render_style;
    }
}

struct LodDistanceWorker {
    distance_tx: mpsc::Sender<f32>,
    resolution_rx: mpsc::Receiver<usize>,
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
    settings: Arc<Mutex<LodSettings>>,
}

impl LodDistanceWorker {
    fn new(settings: Arc<Mutex<LodSettings>>) -> Self {
        let (distance_tx, distance_rx) = mpsc::channel::<f32>();
        let (resolution_tx, resolution_rx) = mpsc::channel::<usize>();
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker_settings = Arc::clone(&settings);

        let handle = thread::spawn(move || {
            let mut latest_distance = None;
            let mut last_resolution = None;

            while !worker_stop.load(Ordering::Relaxed) {
                let interval = {
                    let settings = worker_settings
                        .lock()
                        .ok()
                        .map(|guard| *guard)
                        .unwrap_or_default();
                    let fps = settings.distance_check_fps.max(1.0);
                    Duration::from_secs_f32(1.0 / fps)
                };

                match distance_rx.recv_timeout(interval) {
                    Ok(distance) => {
                        latest_distance = Some(distance);
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {}
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }

                while let Ok(distance) = distance_rx.try_recv() {
                    latest_distance = Some(distance);
                }

                let Some(distance) = latest_distance else {
                    continue;
                };

                let settings = worker_settings
                    .lock()
                    .ok()
                    .map(|guard| *guard)
                    .unwrap_or_default();
                if !settings.enabled {
                    continue;
                }

                let resolution = resolution_for_distance(distance, settings);
                if last_resolution != Some(resolution) {
                    let _ = resolution_tx.send(resolution);
                    last_resolution = Some(resolution);
                }
            }
        });

        Self {
            distance_tx,
            resolution_rx,
            stop,
            handle: Some(handle),
            settings,
        }
    }

    fn submit_distance(&self, distance: f32) {
        let _ = self.distance_tx.send(distance);
    }

    fn set_settings(&self, settings: LodSettings) {
        if let Ok(mut guard) = self.settings.lock() {
            *guard = settings;
        }
    }

    fn poll_resolution(&self) -> Option<usize> {
        let mut latest = None;
        while let Ok(resolution) = self.resolution_rx.try_recv() {
            latest = Some(resolution);
        }
        latest
    }
}

impl Drop for LodDistanceWorker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
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
    color: [f32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CircleQuadVertex {
    corner: [f32; 2],
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

struct GpuResources {
    pipeline: wgpu::RenderPipeline,
    additional_pipeline: wgpu::RenderPipeline,
    wire_pipeline: wgpu::RenderPipeline,
    circles_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    vertex_batches: Vec<VertexBufferBatch>,
    circles_quad_buffer: wgpu::Buffer,
    circles_instance_buffer: Option<wgpu::Buffer>,
    circles_instance_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GeometryCacheKey {
    molecule_ptr: usize,
    render_style: RenderStyle,
    color_fn_ptr: usize,
    mesh_resolution: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BallstickQuality {
    High,
    Medium,
    Low,
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
    lod_worker: LodDistanceWorker,
    additional_renders: Vec<Box<dyn AdditionalRender>>,
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
        let mesh_resolution = preference.mesh_resolution();
        let mesh_resolution = mesh_resolution.max(3);
        let sphere_lat = mesh_resolution;
        let sphere_lon = mesh_resolution * 2;
        let cylinder_sides = DEFAULT_BOND_CYLINDER_SIDES;
        let lod_settings = Arc::new(Mutex::new(preference.lod_settings()));

        Self {
            width: 0,
            height: 0,
            preference,
            color_texture: None,
            depth_texture: None,
            texture_id: None,
            gpu: None,
            sphere_mesh: RenderMesh::new_sphere_uv(1.0, sphere_lat, sphere_lon),
            cylinder_mesh: RenderMesh::new_cylinder_open_ended(1.0, 1.0, cylinder_sides),
            geometry_cache_key: None,
            lod_worker: LodDistanceWorker::new(lod_settings),
            additional_renders: Vec::new(),
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
        self.lod_worker.set_settings(lod_settings);
        self.geometry_cache_key = None;
    }

    pub fn render_style(&self) -> RenderStyle {
        self.preference.render_style()
    }

    pub fn is_low_mode(&self) -> bool {
        self.preference.is_low_mode
    }

    pub fn set_render_style(&mut self, render_style: RenderStyle) {
        if self.preference.render_style() != render_style {
            self.preference.set_render_style(render_style);
            self.geometry_cache_key = None;
        }
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
        self.preference = preference;

        if mesh_resolution_changed {
            let mesh_resolution = self.preference.mesh_resolution();
            let sphere_lat = mesh_resolution;
            let sphere_lon = mesh_resolution * 2;
            let cylinder_sides = DEFAULT_BOND_CYLINDER_SIDES;
            self.sphere_mesh = RenderMesh::new_sphere_uv(1.0, sphere_lat, sphere_lon);
            self.cylinder_mesh = RenderMesh::new_cylinder_open_ended(1.0, 1.0, cylinder_sides);
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

    fn upload_vertex_batches(
        device: &wgpu::Device,
        label_prefix: &str,
        vertices: Vec<Vertex>,
        primitive_stride: usize,
    ) -> Vec<VertexBufferBatch> {
        let ranges = vertex_batch_bounds(vertices.len(), primitive_stride, MAX_RENDER_VERTICES);
        let mut batches = Vec::with_capacity(ranges.len());

        for (batch_index, (start, len)) in ranges.into_iter().enumerate() {
            let end = start + len;
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&format!("{label_prefix}-{batch_index}")),
                contents: bytemuck::cast_slice(&vertices[start..end]),
                usage: wgpu::BufferUsages::VERTEX,
            });

            batches.push(VertexBufferBatch {
                buffer,
                count: len as u32,
            });
        }

        batches
    }

    fn draw_vertex_batches(pass: &mut wgpu::RenderPass<'_>, batches: &[VertexBufferBatch]) {
        for batch in batches {
            pass.set_vertex_buffer(0, batch.buffer.slice(..));
            pass.draw(0..batch.count, 0..1);
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

        let is_circles = matches!(self.preference.render_style(), RenderStyle::Circles);
        let style = if is_circles {
            None
        } else {
            Some(style_for(self.preference.render_style()))
        };
        let cache_key = self.build_geometry_cache_key(frame.molecule, frame.color_fn);
        let rebuilt_vertices = if !is_circles && self.geometry_cache_key != Some(cache_key) {
            Some(self.build_scene_vertices(frame.molecule, frame.color_fn))
        } else {
            None
        };
        let circles_instances = if is_circles {
            Some(build_circle_instances(
                frame.molecule,
                frame.color_fn,
                frame.camera_position,
            ))
        } else {
            None
        };
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

        if let Some(vertices) = rebuilt_vertices {
            let primitive_stride = style
                .map(|active_style| active_style.primitive_stride())
                .unwrap_or(3);

            gpu.vertex_batches = Self::upload_vertex_batches(
                &render_state.device,
                "offscreen-vertex-buffer",
                vertices,
                primitive_stride,
            );
            self.geometry_cache_key = Some(cache_key);
        }

        if let Some(instances) = circles_instances {
            if instances.is_empty() {
                gpu.circles_instance_buffer = None;
                gpu.circles_instance_count = 0;
            } else {
                gpu.circles_instance_count = instances.len() as u32;
                gpu.circles_instance_buffer = Some(render_state.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("offscreen-circles-instance-buffer"),
                        contents: bytemuck::cast_slice(&instances),
                        usage: wgpu::BufferUsages::VERTEX,
                    },
                ));
            }
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

            let pipeline = match self.preference.render_style() {
                RenderStyle::BallStick if self.preference.is_low_mode => &gpu.wire_pipeline,
                RenderStyle::BallStick => &gpu.pipeline,
                RenderStyle::BallOnly => &gpu.pipeline,
                RenderStyle::Circles => &gpu.circles_pipeline,
                RenderStyle::Wireframe => &gpu.wire_pipeline,
            };

            pass.set_pipeline(pipeline);

            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            if is_circles {
                if let Some(instance_buffer) = &gpu.circles_instance_buffer {
                    pass.set_vertex_buffer(0, gpu.circles_quad_buffer.slice(..));
                    pass.set_vertex_buffer(1, instance_buffer.slice(..));
                    pass.draw(0..6, 0..gpu.circles_instance_count);
                }
            } else if !gpu.vertex_batches.is_empty() {
                Self::draw_vertex_batches(&mut pass, &gpu.vertex_batches);
            }

            for (pipeline_kind, additional_vertices, additional_sphere_impostors) in
                additional_batches.into_iter()
            {
                if !additional_vertices.is_empty() && pipeline_kind != GpuPipeline::SphereImpostor {
                    let primitive_stride = match pipeline_kind {
                        GpuPipeline::Triangles => 3,
                        GpuPipeline::Wireframe => 2,
                        GpuPipeline::SphereImpostor => 3,
                    };
                    let additional_vertex_batches = Self::upload_vertex_batches(
                        &render_state.device,
                        "offscreen-additional-vertex-buffer",
                        additional_vertices,
                        primitive_stride,
                    );

                    let pipeline = Self::additional_pipeline_for(gpu, pipeline_kind);
                    pass.set_pipeline(pipeline);
                    Self::draw_vertex_batches(&mut pass, &additional_vertex_batches);
                }

                if !additional_sphere_impostors.is_empty() {
                    let sphere_instance_buffer =
                        render_state
                            .device
                            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                label: Some("offscreen-additional-sphere-instance-buffer"),
                                contents: bytemuck::cast_slice(&additional_sphere_impostors),
                                usage: wgpu::BufferUsages::VERTEX,
                            });

                    let pipeline = Self::additional_pipeline_for(gpu, GpuPipeline::SphereImpostor);
                    pass.set_pipeline(pipeline);
                    pass.set_vertex_buffer(0, gpu.circles_quad_buffer.slice(..));
                    pass.set_vertex_buffer(1, sphere_instance_buffer.slice(..));
                    pass.draw(0..6, 0..additional_sphere_impostors.len() as u32);
                }
            }
        }

        render_state.queue.submit(std::iter::once(encoder.finish()));
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
            self.preference.is_low_mode,
        );
        self.render_frame_with_state(render_state, &frame)
    }

    pub fn submit_lod_distance(&self, distance: f32) {
        self.lod_worker.submit_distance(distance);
    }

    fn apply_pending_lod_resolution(&mut self) {
        let Some(target_resolution) = self.lod_worker.poll_resolution() else {
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

    fn build_geometry_cache_key(
        &self,
        molecule: Option<&Molecule>,
        color_fn: ColorFn,
    ) -> GeometryCacheKey {
        GeometryCacheKey {
            molecule_ptr: molecule
                .map(|mol| mol as *const Molecule as usize)
                .unwrap_or(0),
            render_style: self.preference.render_style(),
            color_fn_ptr: color_fn as usize,
            mesh_resolution: self.preference.mesh_resolution(),
        }
    }

    fn build_scene_vertices(&self, molecule: Option<&Molecule>, color_fn: ColorFn) -> Vec<Vertex> {
        match self.preference.render_style() {
            RenderStyle::BallStick => self.build_ballstick_vertices(molecule, color_fn),
            RenderStyle::BallOnly => self.build_ballstick_vertices(molecule, color_fn),
            RenderStyle::Circles => Vec::new(),
            RenderStyle::Wireframe => self.build_wireframe_vertices(molecule, color_fn),
        }
    }

    fn build_additional_scene_vertices(&self, scene: &Scene) -> Vec<Vertex> {
        let mut vertices = Vec::new();

        for entity in &scene.entities {
            let Some(mesh) = scene.meshes.get(entity.mesh) else {
                continue;
            };

            let scale =
                entity
                    .scale_partial
                    .unwrap_or(Vec3::new(entity.scale, entity.scale, entity.scale));
            let color = [entity.color.0, entity.color.1, entity.color.2];

            for tri in mesh.indices.chunks_exact(3) {
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

    fn build_ballstick_vertices(
        &self,
        molecule: Option<&Molecule>,
        color_fn: ColorFn,
    ) -> Vec<Vertex> {
        let render_style = self.preference.render_style();
        let include_bonds = !matches!(render_style, RenderStyle::BallOnly);
        let quality = self.pick_ballstick_quality(molecule);
        let mesh_resolution = self.preference.mesh_resolution();
        let quality_resolution = match quality {
            BallstickQuality::High => mesh_resolution.max(3),
            BallstickQuality::Medium => (mesh_resolution / 2).max(3),
            BallstickQuality::Low => 3,
        };
        let low_mode = matches!(quality, BallstickQuality::Low);

        let mesh_resolution = self.preference.mesh_resolution();
        let generated_meshes = if quality_resolution == mesh_resolution {
            None
        } else {
            Some((
                RenderMesh::new_sphere_uv(1.0, quality_resolution, quality_resolution * 2),
                RenderMesh::new_cylinder_open_ended(1.0, 1.0, DEFAULT_BOND_CYLINDER_SIDES),
            ))
        };
        let (sphere_mesh, cylinder_mesh): (&RenderMesh, &RenderMesh) =
            if let Some((sphere, cylinder)) = &generated_meshes {
                (sphere, cylinder)
            } else {
                (&self.sphere_mesh, &self.cylinder_mesh)
            };

        let max_vertices = MAX_RENDER_VERTICES;
        let mut vertices = if let Some(mol) = molecule {
            // Estimate capacity: bonds * ~50 vertices + atoms * ~200 vertices + axes * ~75 vertices
            let capacity = mol
                .bonds
                .len()
                .saturating_mul(50)
                .saturating_add(mol.atoms.len().saturating_mul(200))
                .saturating_add(225)
                .min(max_vertices);
            Vec::with_capacity(capacity)
        } else {
            Vec::with_capacity(225.min(max_vertices)) // Just axes
        };

        if let Some(mol) = molecule {
            if include_bonds {
                'bonds: for bond in &mol.bonds {
                    let a = mol.atoms[bond.atom_a].position;
                    let b = mol.atoms[bond.atom_b].position;
                    let diff = b - a;
                    let len = diff.magnitude();
                    if len < 0.001 {
                        continue;
                    }

                    let dir = diff.to_normalized();
                    let up = Vec3::new(0.0, 1.0, 0.0);
                    let orientation = Quaternion::from_unit_vecs(up, dir);
                    let mid = (a + b) * 0.5;

                    let bond_order = bond.order.max(1) as usize;
                    let line_offsets = bond_line_offsets(bond_order);
                    let mut lateral = Vec3::new(1.0, 0.0, 0.0);
                    if dir.dot(lateral).abs() > 0.9 {
                        lateral = Vec3::new(0.0, 0.0, 1.0);
                    }
                    lateral = (lateral - dir * lateral.dot(dir)).to_normalized();

                    let base_radius = if bond_order <= 1 {
                        default_ball_stick_bond_radius()
                    } else {
                        default_ball_stick_bond_radius() * 0.5
                    };
                    for offset in line_offsets {
                        if low_mode {
                            if !append_line(
                                &mut vertices,
                                a + lateral * offset,
                                b + lateral * offset,
                                [0.55, 0.55, 0.55],
                                max_vertices,
                            ) {
                                break 'bonds;
                            }
                        } else if !append_mesh_triangles(
                            &mut vertices,
                            cylinder_mesh,
                            mid + lateral * offset,
                            orientation,
                            Vec3::new(base_radius, len, base_radius),
                            [0.55, 0.55, 0.55],
                            max_vertices,
                        ) {
                            break 'bonds;
                        }
                    }
                }
            }

            'atoms: for (_idx, atom) in mol.atoms.iter().enumerate() {
                let pos = atom.position;
                let radius = ball_stick_radius(&atom.element, false);
                let color_tuple = color_fn(atom, false);
                let color = [color_tuple.0, color_tuple.1, color_tuple.2];

                if !append_mesh_triangles(
                    &mut vertices,
                    sphere_mesh,
                    pos,
                    Quaternion::new_identity(),
                    Vec3::new(radius, radius, radius),
                    color,
                    max_vertices,
                ) {
                    break 'atoms;
                }
            }
        }

        if include_bonds {
            // xyz axes as cylinders
            let axis_len = 0.2;
            let axis_radius = 0.01;
            if low_mode {
                let _ = append_line(
                    &mut vertices,
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(axis_len, 0.0, 0.0),
                    [1.0, 0.0, 0.0],
                    max_vertices,
                );
                let _ = append_line(
                    &mut vertices,
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, axis_len, 0.0),
                    [0.0, 1.0, 0.0],
                    max_vertices,
                );
                let _ = append_line(
                    &mut vertices,
                    Vec3::new(0.0, 0.0, 0.0),
                    Vec3::new(0.0, 0.0, axis_len),
                    [0.0, 0.0, 1.0],
                    max_vertices,
                );
            } else {
                let _ = append_mesh_triangles(
                    &mut vertices,
                    cylinder_mesh,
                    Vec3::new(axis_len * 0.5, 0.0, 0.0),
                    Quaternion::from_axis_angle(
                        Vec3::new(0.0, 0.0, 1.0),
                        -std::f32::consts::FRAC_PI_2,
                    ),
                    Vec3::new(axis_radius, axis_len, axis_radius),
                    [1.0, 0.0, 0.0],
                    max_vertices,
                );
                let _ = append_mesh_triangles(
                    &mut vertices,
                    cylinder_mesh,
                    Vec3::new(0.0, axis_len * 0.5, 0.0),
                    Quaternion::new_identity(),
                    Vec3::new(axis_radius, axis_len, axis_radius),
                    [0.0, 1.0, 0.0],
                    max_vertices,
                );
                let _ = append_mesh_triangles(
                    &mut vertices,
                    cylinder_mesh,
                    Vec3::new(0.0, 0.0, axis_len * 0.5),
                    Quaternion::from_axis_angle(
                        Vec3::new(1.0, 0.0, 0.0),
                        std::f32::consts::FRAC_PI_2,
                    ),
                    Vec3::new(axis_radius, axis_len, axis_radius),
                    [0.0, 0.0, 1.0],
                    max_vertices,
                );
            }
        }

        vertices
    }

    fn pick_ballstick_quality(&self, molecule: Option<&Molecule>) -> BallstickQuality {
        let Some(mol) = molecule else {
            return BallstickQuality::High;
        };

        let high = self.estimate_ballstick_vertices(mol, BallstickQuality::High);
        if high <= MAX_RENDER_VERTICES {
            return BallstickQuality::High;
        }

        let medium = self.estimate_ballstick_vertices(mol, BallstickQuality::Medium);
        if medium <= MAX_RENDER_VERTICES {
            return BallstickQuality::Medium;
        }

        BallstickQuality::Low
    }

    fn estimate_ballstick_vertices(&self, molecule: &Molecule, quality: BallstickQuality) -> usize {
        let resolution = match quality {
            BallstickQuality::High => self.preference.mesh_resolution().max(3),
            BallstickQuality::Medium => (self.preference.mesh_resolution() / 2).max(3),
            BallstickQuality::Low => 3,
        };

        let sphere_vertices_per_atom = resolution
            .saturating_mul(resolution.saturating_mul(2))
            .saturating_mul(6);
        let atom_vertices = molecule
            .atoms
            .len()
            .saturating_mul(sphere_vertices_per_atom);

        let bond_vertices = if matches!(quality, BallstickQuality::Low) {
            molecule.bonds.len().saturating_mul(2)
        } else {
            let cylinder_vertices = DEFAULT_BOND_CYLINDER_SIDES.saturating_mul(6);
            let bond_instances = molecule.bonds.iter().fold(0usize, |acc, bond| {
                acc.saturating_add(bond_line_offsets(bond.order.max(1) as usize).len())
            });
            bond_instances.saturating_mul(cylinder_vertices)
        };

        let axis_vertices = if matches!(quality, BallstickQuality::Low) {
            6
        } else {
            (resolution.saturating_mul(2))
                .saturating_mul(12)
                .saturating_mul(3)
        };

        atom_vertices
            .saturating_add(bond_vertices)
            .saturating_add(axis_vertices)
    }

    fn build_wireframe_vertices(
        &self,
        molecule: Option<&Molecule>,
        color_fn: ColorFn,
    ) -> Vec<Vertex> {
        let max_vertices = MAX_RENDER_VERTICES;
        let mut vertices = if let Some(mol) = molecule {
            // Estimate capacity: bonds * ~4 + atoms * ~6 + axes * ~6
            // This is much less than ballstick since we're only adding lines
            let capacity = mol
                .bonds
                .len()
                .saturating_mul(4)
                .saturating_add(mol.atoms.len().saturating_mul(6))
                .saturating_add(6)
                .min(max_vertices);
            Vec::with_capacity(capacity)
        } else {
            Vec::with_capacity(6.min(max_vertices)) // Just axes
        };

        if let Some(mol) = molecule {
            'bonds: for bond in &mol.bonds {
                let a = mol.atoms[bond.atom_a].position;
                let b = mol.atoms[bond.atom_b].position;
                let diff = b - a;
                let len = diff.magnitude();
                if len < 0.001 {
                    continue;
                }

                let dir = diff.to_normalized();
                let mut lateral = Vec3::new(1.0, 0.0, 0.0);
                if dir.dot(lateral).abs() > 0.9 {
                    lateral = Vec3::new(0.0, 0.0, 1.0);
                }
                lateral = (lateral - dir * lateral.dot(dir)).to_normalized();

                let bond_order = bond.order.max(1) as usize;
                for offset in bond_line_offsets(bond_order) {
                    let off = lateral * offset;
                    if !append_line(
                        &mut vertices,
                        a + off,
                        b + off,
                        [0.70, 0.70, 0.72],
                        max_vertices,
                    ) {
                        break 'bonds;
                    }
                }
            }

            'atoms: for (_idx, atom) in mol.atoms.iter().enumerate() {
                let pos = atom.position;
                let span = 0.02;
                let color_tuple = color_fn(atom, false);
                let color = [color_tuple.0, color_tuple.1, color_tuple.2];

                if !append_line(
                    &mut vertices,
                    pos + Vec3::new(-span, 0.0, 0.0),
                    pos + Vec3::new(span, 0.0, 0.0),
                    color,
                    max_vertices,
                ) {
                    break 'atoms;
                }
                if !append_line(
                    &mut vertices,
                    pos + Vec3::new(0.0, -span, 0.0),
                    pos + Vec3::new(0.0, span, 0.0),
                    color,
                    max_vertices,
                ) {
                    break 'atoms;
                }
                if !append_line(
                    &mut vertices,
                    pos + Vec3::new(0.0, 0.0, -span),
                    pos + Vec3::new(0.0, 0.0, span),
                    color,
                    max_vertices,
                ) {
                    break 'atoms;
                }
            }
        }

        let axis_len = 2.0;
        let _ = append_line(
            &mut vertices,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(axis_len, 0.0, 0.0),
            [1.0, 0.0, 0.0],
            max_vertices,
        );
        let _ = append_line(
            &mut vertices,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, axis_len, 0.0),
            [0.0, 1.0, 0.0],
            max_vertices,
        );
        let _ = append_line(
            &mut vertices,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, axis_len),
            [0.0, 0.0, 1.0],
            max_vertices,
        );

        vertices
    }
}

impl Default for OffscreenRenderer {
    fn default() -> Self {
        Self::new()
    }
}

fn create_gpu_resources(device: &wgpu::Device) -> GpuResources {
    let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("offscreen-uniform-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let initial_uniforms = Uniforms {
        view_proj: [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        viewport: [1.0, 1.0],
        focal: 1.0,
        _pad: 0.0,
        camera_right: [1.0, 0.0, 0.0, 0.0],
        camera_up: [0.0, 1.0, 0.0, 0.0],
        camera_forward: [0.0, 0.0, 1.0, 0.0],
    };
    let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offscreen-uniform-buffer"),
        contents: bytemuck::bytes_of(&initial_uniforms),
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
    });
    let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("offscreen-uniform-bind-group"),
        layout: &uniform_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("offscreen-shader"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) depth01: f32,
};

struct Uniforms {
    view_proj: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: f32,
    _pad: f32,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    camera_forward: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
) -> VSOut {
    var out: VSOut;
    let clip = uniforms.view_proj * vec4<f32>(position, 1.0);
    out.position = clip;
    out.color = color;
    out.normal = normal;
    out.depth01 = clip.z / clip.w * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(vec3<f32>(0.35, 0.75, 0.55));
    let diffuse = max(dot(n, l), 0.0);
    let ambient = 0.22;
    let standard_lit = in.color * (ambient + 0.78 * diffuse);

    // For deeper fragments, blend toward a sphere-like radial tone.
    let depth01 = clamp(in.depth01, 0.0, 1.0);
    let deep_factor = smoothstep(0.60, 0.98, depth01);
    let radial = clamp(n.z * 0.5 + 0.5, 0.0, 1.0);

    let core = in.color * 1.05 + vec3<f32>(0.10, 0.10, 0.10);
    let mid = in.color;
    let edge = in.color * 0.28;
    let center_mix = smoothstep(0.55, 1.0, radial);
    let radial_mix = smoothstep(0.05, 0.85, radial);
    var sphere_tone = mix(edge, mix(mid, core, center_mix), radial_mix);

    // Darker rim to emulate the sample's white->base->black feel.
    let rim = mix(0.72, 1.0, smoothstep(0.0, 0.55, radial));
    sphere_tone = sphere_tone * rim;

    let lit = mix(standard_lit, sphere_tone, deep_factor);
    return vec4<f32>(lit, 1.0);
}
"#
            .into(),
        ),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("offscreen-layout"),
        bind_group_layouts: &[Some(&uniform_layout)],
        immediate_size: 0,
    });

    let pipeline = create_triangle_pipeline(device, &layout, &shader, "offscreen-pipeline");
    let additional_pipeline =
        create_triangle_pipeline(device, &layout, &shader, "offscreen-additional-pipeline");

    let wire_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("offscreen-wire-shader"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

struct Uniforms {
    view_proj: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: f32,
    _pad: f32,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    camera_forward: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) _normal: vec3<f32>,
    @location(2) color: vec3<f32>,
) -> VSOut {
    var out: VSOut;
    out.position = uniforms.view_proj * vec4<f32>(position, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    return vec4<f32>(in.color, 1.0);
}
"#
            .into(),
        ),
    });

    let wire_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("offscreen-wire-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &wire_shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: std::mem::size_of::<[f32; 3]>() as u64,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: (std::mem::size_of::<[f32; 3]>() * 2) as u64,
                        shader_location: 2,
                    },
                ],
            }],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &wire_shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::LineList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    });

    let circles_pipeline = create_circles_pipeline(device, &layout);
    let circles_quad_vertices = [
        CircleQuadVertex {
            corner: [-1.0, -1.0],
        },
        CircleQuadVertex {
            corner: [1.0, -1.0],
        },
        CircleQuadVertex { corner: [1.0, 1.0] },
        CircleQuadVertex {
            corner: [-1.0, -1.0],
        },
        CircleQuadVertex { corner: [1.0, 1.0] },
        CircleQuadVertex {
            corner: [-1.0, 1.0],
        },
    ];
    let circles_quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offscreen-circles-quad-buffer"),
        contents: bytemuck::cast_slice(&circles_quad_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    GpuResources {
        pipeline,
        additional_pipeline,
        wire_pipeline,
        circles_pipeline,
        uniform_buffer,
        uniform_bind_group,
        vertex_batches: Vec::new(),
        circles_quad_buffer,
        circles_instance_buffer: None,
        circles_instance_count: 0,
    }
}

fn create_triangle_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<Vertex>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &[
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: 0,
                        shader_location: 0,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: std::mem::size_of::<[f32; 3]>() as u64,
                        shader_location: 1,
                    },
                    wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x3,
                        offset: (std::mem::size_of::<[f32; 3]>() * 2) as u64,
                        shader_location: 2,
                    },
                ],
            }],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::REPLACE),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: Some(wgpu::Face::Back),
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_circles_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("offscreen-circles-shader"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
struct Uniforms {
    view_proj: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: f32,
    _pad: f32,
    camera_right: vec4<f32>,
    camera_up: vec4<f32>,
    camera_forward: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

struct VSOut {
    @builtin(position) position: vec4<f32>,

    // sphere center in world/view space
    @location(0) center: vec3<f32>,

    @location(1) radius: f32,
    @location(2) color: vec3<f32>,
    @location(3) local: vec2<f32>,
};

struct FSOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@vertex
fn vs_main(
    @location(0) corner: vec2<f32>,
    @location(1) center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec3<f32>,
) -> VSOut {
    var out: VSOut;

    let clip_center =
        uniforms.view_proj * vec4<f32>(center, 1.0);

    let inv_w =
        1.0 / max(abs(clip_center.w), 0.0001);

    let ndc_radius =
        radius
        * uniforms.focal
        * inv_w;

    let aspect =
        uniforms.viewport.x
        / max(uniforms.viewport.y, 0.0001);

    // clip-space billboard offset
    let corner_clip =
        vec2<f32>(
            corner.x / aspect,
            corner.y,
        );

    let clip_offset =
        corner_clip
        * ndc_radius
        * clip_center.w;

    out.position = vec4<f32>(
        clip_center.xy + clip_offset,
        clip_center.z,
        clip_center.w,
    );

    out.center = center;
    out.radius = radius;
    out.color = color;

    // IMPORTANT:
    // local coordinates must remain unit-circle space
    // DO NOT apply aspect correction here
    out.local = corner;

    return out;
}

@fragment
fn fs_main(in: VSOut) -> FSOut {
    var out: FSOut;

    let xy = in.local;

    let r2 = dot(xy, xy);

    if (r2 > 1.0) {
        discard;
    }

    let z = sqrt(1.0 - r2);

    let normal =
        vec3<f32>(xy, z);

    let cam_right = uniforms.camera_right.xyz;
    let cam_up = uniforms.camera_up.xyz;
    let cam_forward = uniforms.camera_forward.xyz;

    let sphere_offset =
        cam_right * normal.x
        + cam_up * normal.y
        - cam_forward * normal.z;

    let world_normal =
        normalize(sphere_offset);

    let sphere_pos =
        in.center + world_normal * in.radius;

    let clip =
        uniforms.view_proj * vec4<f32>(sphere_pos, 1.0);

    out.depth =
        clip.z / clip.w * 0.5 + 0.5;

    let light_dir =
        normalize(vec3<f32>(0.3, 0.5, 1.0));

    let diffuse =
        max(dot(world_normal, light_dir), 0.0);

    let ambient = 0.15;

    let lit =
        in.color * (ambient + diffuse * 0.85);

    out.color = vec4<f32>(lit, 1.0);

    return out;
}
"#
            .into(),
        ),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("offscreen-circles-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_main"),
            buffers: &[
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<CircleQuadVertex>() as u64,
                    step_mode: wgpu::VertexStepMode::Vertex,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Float32x2,
                        offset: 0,
                        shader_location: 0,
                    }],
                },
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<CircleInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 1,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: std::mem::size_of::<[f32; 3]>() as u64,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: std::mem::size_of::<[f32; 4]>() as u64,
                            shader_location: 3,
                        },
                    ],
                },
            ],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: wgpu::TextureFormat::Depth24Plus,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
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
    position: Vec3,
    orientation: Quaternion,
    scale: Vec3,
    color: [f32; 3],
    max_vertices: usize,
) -> bool {
    let inv_scale = Vec3::new(
        if scale.x.abs() > 1e-6 {
            1.0 / scale.x
        } else {
            0.0
        },
        if scale.y.abs() > 1e-6 {
            1.0 / scale.y
        } else {
            0.0
        },
        if scale.z.abs() > 1e-6 {
            1.0 / scale.z
        } else {
            0.0
        },
    );

    for tri in mesh.indices.chunks_exact(3) {
        if out.len().saturating_add(3) > max_vertices {
            return false;
        }

        for &idx in tri {
            let Some(src) = mesh.vertices.get(idx) else {
                return false;
            };

            let p = Vec3::new(src.position[0], src.position[1], src.position[2]);
            let p_scaled = Vec3::new(p.x * scale.x, p.y * scale.y, p.z * scale.z);
            let p_world = orientation.rotate_vec(p_scaled) + position;

            let n = Vec3::new(src.normal[0], src.normal[1], src.normal[2]);
            let n_scaled = Vec3::new(n.x * inv_scale.x, n.y * inv_scale.y, n.z * inv_scale.z);
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
        let index_count = lat_segments * lon_segments * 6; // 2 triangles per quad

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
                let n = Vec3::new(x, y, z).to_normalized();

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
        // Pre-calculate capacities
        let vertex_capacity = sides * 2; // Side vertices only
        let index_capacity = sides * 6; // Side quads (2 triangles each)

        let mut vertices = Vec::with_capacity(vertex_capacity);
        let mut indices = Vec::with_capacity(index_capacity);
        let half = len * 0.5;

        for i in 0..sides {
            let t = i as f32 / sides as f32;
            let angle = t * std::f32::consts::TAU;
            let x = radius * angle.cos();
            let z = radius * angle.sin();
            let n = Vec3::new(x, 0.0, z).to_normalized();

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
    a: Vec3,
    b: Vec3,
    color: [f32; 3],
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

fn resolution_for_distance(distance: f32, lod_settings: LodSettings) -> usize {
    if distance <= lod_settings.high_detail_max_distance {
        lod_settings.high_detail_mesh_resolution
    } else if distance <= lod_settings.medium_detail_max_distance {
        lod_settings.medium_detail_mesh_resolution
    } else {
        lod_settings.low_detail_mesh_resolution
    }
}

fn vertex_batch_bounds(
    total_vertices: usize,
    primitive_stride: usize,
    batch_vertex_limit: usize,
) -> Vec<(usize, usize)> {
    let primitive_stride = primitive_stride.max(1);
    let mut batch_vertex_limit = batch_vertex_limit.max(primitive_stride);
    batch_vertex_limit -= batch_vertex_limit % primitive_stride;
    if batch_vertex_limit == 0 {
        batch_vertex_limit = primitive_stride;
    }

    let usable_vertices = total_vertices - (total_vertices % primitive_stride);
    let mut ranges = Vec::new();
    let mut start = 0;

    while start < usable_vertices {
        let remaining = usable_vertices - start;
        let len = remaining.min(batch_vertex_limit);
        ranges.push((start, len));
        start += len;
    }

    ranges
}

#[cfg(test)]
mod tests {
    use super::vertex_batch_bounds;

    #[test]
    fn vertex_batch_bounds_keeps_full_primitives() {
        assert_eq!(
            vertex_batch_bounds(12, 3, 5),
            vec![(0, 3), (3, 3), (6, 3), (9, 3)]
        );
    }

    #[test]
    fn vertex_batch_bounds_drops_incomplete_tail() {
        assert_eq!(vertex_batch_bounds(10, 3, 5), vec![(0, 3), (3, 3), (6, 3)]);
    }
}
