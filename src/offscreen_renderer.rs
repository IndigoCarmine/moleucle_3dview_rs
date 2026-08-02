use crate::additional_render::{AdditionalRender, GpuPipeline};
use crate::frame_state::RenderFrameState;
use crate::render_state::SharedRenderStates;
use crate::scene_types::Scene;
use crate::viewer::ColorFn;
use crate::Molecule;
use egui::TextureId;
use egui_wgpu::wgpu;
use lin_alg::f32::Vec3;

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

/// Identifies the geometry currently uploaded to the GPU, so an unchanged frame
/// costs a uniform write and the draw calls, with no CPU rebuild or upload.
///
/// Everything about the *molecule* — which molecule, its positions, its colors,
/// its per-atom overrides — is folded into `geometry_revision`
/// ([`RenderFrameState::geometry_revision`]). The two render settings are
/// tracked separately because [`OffscreenRenderer::apply_pending_lod_resolution`]
/// can change the mesh resolution at the top of a render, after the caller has
/// already assembled the frame.
///
/// This deliberately does *not* key on the molecule's address. It used to, and
/// that was the bug: `set_molecule` writes into the viewer's existing `Option`
/// slot, so replacing the molecule left the pointer unchanged and the cache hit
/// against geometry for a molecule that no longer existed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GeometryCacheKey {
    geometry_revision: u64,
    render_style: RenderStyle,
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
    /// While set, [`OffscreenRenderer::apply_pending_lod_resolution`] is a no-op
    /// so a caller-forced mesh resolution survives the frame.
    lod_locked: bool,
    additional_renders: Vec<Box<dyn AdditionalRender>>,
    /// Per-overlay CPU scratch and GPU buffers, index-aligned with
    /// `additional_renders`. Kept across frames so an unchanged overlay costs a
    /// `write_buffer` into storage it already owns instead of a fresh
    /// allocation on both sides.
    additional_batches: Vec<AdditionalBatch>,
    /// Reusable CPU scratch buffers for impostor/bond instances, kept across
    /// frames so trajectory playback refills them without reallocating.
    scratch_circle_instances: Vec<CircleInstance>,
    scratch_bond_instances: Vec<BondInstance>,
}

/// One [`AdditionalRender`]'s retained CPU scratch and GPU buffers.
///
/// Overlay geometry used to be rebuilt into fresh `Vec`s and uploaded to a
/// brand-new `wgpu::Buffer` created *inside the render pass*, every batch, every
/// frame. With several overlays registered that is a steady stream of buffer
/// creation and CPU allocation on the render path; the main molecule has had a
/// cache and reusable scratch for exactly this reason.
#[derive(Default)]
struct AdditionalBatch {
    /// Reused across frames; cleared and refilled by `update_scene`.
    scene: Scene,
    /// The scene's entities baked into world-space triangles.
    vertices: Vec<Vertex>,
    pipeline: GpuPipeline,
    vertex_buffer: Option<wgpu::Buffer>,
    vertex_capacity: usize,
    vertex_count: u32,
    sphere_buffer: Option<wgpu::Buffer>,
    sphere_capacity: usize,
    sphere_count: u32,
    /// Whether this batch has to draw in the translucent phase. Computed once
    /// when the batch is rebuilt rather than by rescanning every vertex and
    /// every impostor on every frame.
    translucent: bool,
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
            lod_locked: false,
            preference,
            color_texture: None,
            depth_texture: None,
            texture_id: None,
            gpu: None,
            sphere_mesh: RenderMesh::new_sphere_uv(1.0, sphere_lat, sphere_lon),
            cylinder_mesh: RenderMesh::new_cylinder_open_ended(1.0, 1.0, cylinder_sides),
            geometry_cache_key: None,
            additional_renders: Vec::new(),
            additional_batches: Vec::new(),
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

    /// Current color-target size in pixels, or `(0, 0)` before the first
    /// [`Self::ensure_resources`].
    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Copy the color target back to the CPU as tightly packed, top-down RGBA8
    /// rows (`width * height * 4` bytes).
    ///
    /// The bytes are the raw target contents, which are **premultiplied** wherever
    /// anything translucent was drawn: the pipelines blend with
    /// [`wgpu::BlendState::ALPHA_BLENDING`], so a fragment of opacity `a` over a
    /// transparent clear lands as `rgb = a * color`, `alpha = a`. Callers writing
    /// a PNG have to divide the color through by alpha first — that is what
    /// [`crate::InteractiveMoleculeViewport::render_image`] does. With an opaque
    /// background every pixel comes back at `alpha = 1` and the distinction
    /// disappears.
    ///
    /// Blocks until the GPU finishes the copy, so call it for an explicit export
    /// rather than per frame.
    pub fn read_rgba(&self, render_state: &egui_wgpu::RenderState) -> Result<Vec<u8>, String> {
        let texture = self
            .color_texture
            .as_ref()
            .ok_or_else(|| "No color target to read; render a frame first".to_string())?;
        let (width, height) = (self.width, self.height);
        if width == 0 || height == 0 {
            return Err("Color target has zero size".to_string());
        }

        // `copy_texture_to_buffer` requires each row to start on a 256-byte
        // boundary, so the staging buffer is padded and the rows are compacted
        // again after mapping.
        let unpadded = width * 4;
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded = unpadded.div_ceil(align) * align;

        let device = &render_state.device;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("offscreen-readback"),
            size: (padded as u64) * (height as u64),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("offscreen-readback-encoder"),
        });
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(height),
                },
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        let submission = render_state.queue.submit(std::iter::once(encoder.finish()));

        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|e| format!("Readback poll failed: {e}"))?;

        let mut out = Vec::with_capacity((unpadded as usize) * (height as usize));
        {
            let mapped = slice.get_mapped_range();
            for row in 0..height as usize {
                let start = row * padded as usize;
                out.extend_from_slice(&mapped[start..start + unpadded as usize]);
            }
        }
        staging.unmap();

        Ok(out)
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
        // Keep the retained batches index-aligned with the overlays.
        self.additional_batches.push(AdditionalBatch::default());
    }

    fn additional_pipeline_for(
        gpu: &GpuResources,
        pipeline: GpuPipeline,
        depth_write: bool,
    ) -> &wgpu::RenderPipeline {
        match pipeline {
            GpuPipeline::Triangles => gpu.additional_pipeline.get(depth_write),
            GpuPipeline::Wireframe => gpu.wire_pipeline.get(depth_write),
            GpuPipeline::SphereImpostor => gpu.circles_pipeline.get(depth_write),
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
            style_for(render_style)
        };

        let cache_key = self.build_geometry_cache_key(frame);
        let rebuilt_vertices = if !use_impostors && self.geometry_cache_key != Some(cache_key) {
            if let Some(active_style) = style {
                let ctx = StyleBuildContext {
                    preference: self.preference,
                    sphere_mesh: &self.sphere_mesh,
                    cylinder_mesh: &self.cylinder_mesh,
                    molecule_opacity: frame.molecule_opacity,
                    atom_radii: frame.atom_radii,
                    atom_colors: frame.atom_colors,
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

        // Rebuild every overlay's CPU geometry into its retained scratch. The
        // buffers keep their capacity across frames, so a steady overlay costs
        // no allocation here.
        //
        // `additional_batches` is a direct field rather than something reached
        // through a `&mut self` method on purpose: the render pass below holds
        // `self.gpu` mutably while still reading `self.preference`, and only
        // disjoint field borrows make that legal.
        for (index, additional) in self.additional_renders.iter().enumerate() {
            let batch = &mut self.additional_batches[index];
            batch.scene.clear();
            additional.update_scene(&mut batch.scene, frame);
            batch.pipeline = additional.gpu_pipeline();

            batch.vertices.clear();
            append_scene_triangles(&batch.scene, &mut batch.vertices, MAX_RENDER_VERTICES);

            // Decide the draw phase once, here, instead of rescanning every
            // vertex and impostor of every batch on every frame.
            batch.translucent = batch
                .scene
                .sphere_impostors
                .iter()
                .any(|instance| instance.color[3] < 1.0)
                || batch.vertices.iter().any(|v| v.color[3] < 1.0);
        }

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

        // Upload the overlay batches before the pass opens. `upload_instances`
        // reuses the existing buffer whenever the new data fits, so a steady
        // overlay costs one `write_buffer` instead of creating a buffer per
        // batch per frame from inside the render pass.
        for batch in &mut self.additional_batches {
            batch.vertex_count = if batch.vertices.is_empty()
                || batch.pipeline == GpuPipeline::SphereImpostor
            {
                0
            } else {
                upload_instances(
                    &render_state.device,
                    &render_state.queue,
                    &mut batch.vertex_buffer,
                    &mut batch.vertex_capacity,
                    "offscreen-additional-vertex-buffer",
                    &batch.vertices,
                )
            };

            batch.sphere_count = if batch.scene.sphere_impostors.is_empty() {
                0
            } else {
                upload_instances(
                    &render_state.device,
                    &render_state.queue,
                    &mut batch.sphere_buffer,
                    &mut batch.sphere_capacity,
                    "offscreen-additional-sphere-instance-buffer",
                    &batch.scene.sphere_impostors,
                )
            };
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
                            r: frame.clear_color[0] as f64,
                            g: frame.clear_color[1] as f64,
                            b: frame.clear_color[2] as f64,
                            a: frame.clear_color[3] as f64,
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

            // A faded molecule must not stamp the depth buffer, or the atoms
            // behind it (and any additional render drawn later in this pass)
            // stay depth-culled no matter how low the alpha goes. Per-atom
            // colors carry their own alpha, so check those too.
            let molecule_translucent = frame.molecule_opacity < 1.0
                || frame
                    .atom_colors
                    .is_some_and(|colors| colors.iter().any(|color| color[3] < 1.0));

            // Draw every opaque thing before every translucent thing.
            //
            // Both groups depth-*test*, only the opaque group depth-*writes*, so
            // this is the usual ordering for mixed geometry: the opaque pass
            // establishes the depth buffer, then translucent fragments are
            // correctly hidden where opaque geometry is in front of them and
            // blended over it where they are in front.
            //
            // Drawing in registration order instead let an opaque batch that
            // came later overwrite translucent geometry sitting *in front* of it,
            // because the translucent draw had written no depth to reject it —
            // a faded molecule in front of an opaque NDX group simply vanished.
            //
            // Translucent-vs-translucent is still draw-order dependent; this
            // renderer does no depth sorting and no order-independent
            // transparency.
            // Every pipeline in this pass reads the same uniforms at group 0, so
            // bind once up front. It has to be outside the phase loop below:
            // when the molecule is translucent the opaque phase skips its draw
            // block entirely, and binding in there left the opaque additional
            // batches with no bind group at all (a wgpu validation failure).
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);

            for translucent_phase in [false, true] {
                if molecule_translucent == translucent_phase {
                    let depth_write = !molecule_translucent;

                    let pipeline = if use_impostors {
                        gpu.circles_pipeline.get(depth_write)
                    } else {
                        match self.preference.render_style() {
                            RenderStyle::BallStick if self.preference.is_low_mode() => {
                                gpu.wire_pipeline.get(depth_write)
                            }
                            RenderStyle::BallStick => gpu.pipeline.get(depth_write),
                            RenderStyle::BallOnly => gpu.pipeline.get(depth_write),
                            RenderStyle::Circles => gpu.circles_pipeline.get(depth_write),
                            RenderStyle::Wireframe => gpu.wire_pipeline.get(depth_write),
                        }
                    };

                    pass.set_pipeline(pipeline);

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
                                    pass.set_pipeline(gpu.bond_pipeline.get(depth_write));
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
                }

                for batch in &self.additional_batches {
                    if batch.translucent != translucent_phase {
                        continue;
                    }
                    let depth_write = !batch.translucent;

                    if batch.vertex_count > 0 {
                        if let Some(vertex_buffer) = &batch.vertex_buffer {
                            let pipeline =
                                Self::additional_pipeline_for(gpu, batch.pipeline, depth_write);
                            pass.set_pipeline(pipeline);
                            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                            pass.draw(0..batch.vertex_count, 0..1);
                        }
                    }

                    if batch.sphere_count > 0 {
                        if let Some(sphere_buffer) = &batch.sphere_buffer {
                            let pipeline = Self::additional_pipeline_for(
                                gpu,
                                GpuPipeline::SphereImpostor,
                                depth_write,
                            );
                            pass.set_pipeline(pipeline);
                            pass.set_vertex_buffer(0, gpu.circles_quad_buffer.slice(..));
                            pass.set_vertex_buffer(1, sphere_buffer.slice(..));
                            pass.draw(0..6, 0..batch.sphere_count);
                        }
                    }
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

    /// Stop the LOD manager from changing the mesh resolution.
    ///
    /// Image export forces a high resolution for one frame, and
    /// [`Self::apply_pending_lod_resolution`] runs at the top of every render —
    /// so without this, a queued LOD decision would silently undo it and the
    /// export would come out faceted.
    pub fn set_lod_locked(&mut self, locked: bool) {
        self.lod_locked = locked;
    }

    fn apply_pending_lod_resolution(&mut self) {
        if self.lod_locked {
            return;
        }
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
            // COPY_SRC so `read_rgba` can pull the frame back to the CPU for
            // image export. Without it the copy fails wgpu validation.
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
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
        // Same bond color the mesh BallStick path bakes in, opacity included, so
        // the fallback fades with the molecule instead of staying opaque while
        // its own pipeline has depth writes disabled.
        let color = [0.55, 0.55, 0.55, frame.molecule_opacity];
        out.reserve(mol.bonds.len().min(MAX_IMPOSTOR_INSTANCES));

        for bond in &mol.bonds {
            if out.len() >= MAX_IMPOSTOR_INSTANCES {
                break;
            }
            let Some((a, b)) = mol.bond_endpoints(bond) else {
                continue;
            };
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
        let radii = frame.atom_radii;
        let colors = frame.atom_colors;
        match render_style {
            RenderStyle::Circles => {
                fill_circle_instances(out, frame.molecule, frame.color_fn, opacity, radii, colors)
            }
            RenderStyle::BallOnly => fill_sphere_instances(
                out,
                frame.molecule,
                frame.color_fn,
                opacity,
                radii,
                colors,
                |atom| vdw_radius(&atom.element),
            ),
            // BallStick falls back here only when the mesh path would overflow;
            // bonds are dropped at that scale, but every atom is still shown.
            RenderStyle::BallStick => fill_sphere_instances(
                out,
                frame.molecule,
                frame.color_fn,
                opacity,
                radii,
                colors,
                |atom| ball_stick_radius(&atom.element, false),
            ),
            RenderStyle::Wireframe => out.clear(),
        }
    }

    fn build_geometry_cache_key(&self, frame: &RenderFrameState<'_>) -> GeometryCacheKey {
        GeometryCacheKey {
            geometry_revision: frame.geometry_revision,
            render_style: self.preference.render_style(),
            mesh_resolution: self.preference.mesh_resolution(),
        }
    }

}

/// Bake a [`Scene`]'s entities into world-space triangles, appending to `out`.
///
/// Stops early once `max_vertices` is reached rather than growing without
/// bound; the caller clears `out` first, so its capacity is reused frame to
/// frame.
fn append_scene_triangles(scene: &Scene, out: &mut Vec<Vertex>, max_vertices: usize) {
    for entity in &scene.entities {
        let Some(mesh) = scene.meshes.get(entity.mesh) else {
            continue;
        };

        let scale = entity
            .scale_partial
            .unwrap_or(Vec3::new(entity.scale, entity.scale, entity.scale));
        let inv_scale = inverse_scale(scale);
        // Fold the entity's opacity into the color's alpha channel so both
        // per-color alpha and the entity-wide opacity affect blending.
        let color = [
            entity.color.0,
            entity.color.1,
            entity.color.2,
            entity.color.3 * entity.opacity,
        ];

        for tri in mesh.indices.chunks_exact(3) {
            if out.len().saturating_add(3) > max_vertices {
                return;
            }

            for &idx in tri {
                let Some(src) = mesh.vertices.get(idx) else {
                    return;
                };

                let position = Vec3::new(src.position[0], src.position[1], src.position[2]);
                out.push(transform_vertex(
                    position,
                    src.normal,
                    entity.position,
                    entity.orientation,
                    scale,
                    inv_scale,
                    color,
                ));
            }
        }
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

/// Componentwise reciprocal of a scale, with zero components mapped to zero.
///
/// Normals transform by the inverse transpose of the model matrix. For a
/// scale-then-rotate transform that reduces to scaling the normal by the
/// reciprocal and then applying the same rotation.
#[inline]
fn inverse_scale(scale: Vec3) -> Vec3 {
    let safe = |v: f32| if v.abs() > 1e-6 { 1.0 / v } else { 0.0 };
    Vec3::new(safe(scale.x), safe(scale.y), safe(scale.z))
}

/// Place one mesh vertex in world space.
///
/// Both the molecule geometry and the overlay geometry go through here, which is
/// the point: the overlay path used to rotate the normal without the
/// inverse-scale correction. Every `add_cylinder` sets a `scale_partial` of
/// `(radius, length, radius)` — extremely non-uniform — so overlay cylinders
/// were lit as though they were unscaled, giving axis triads and interaction
/// sticks a flat, banded look that no amount of tweaking the light would fix.
#[inline]
#[allow(clippy::too_many_arguments)]
fn transform_vertex(
    position: Vec3,
    normal: Vec3,
    entity_position: Vec3,
    orientation: lin_alg::f32::Quaternion,
    scale: Vec3,
    inv_scale: Vec3,
    color: [f32; 4],
) -> Vertex {
    let scaled = Vec3::new(
        position.x * scale.x,
        position.y * scale.y,
        position.z * scale.z,
    );
    let world = orientation.rotate_vec(scaled) + entity_position;

    let n_scaled = Vec3::new(
        normal.x * inv_scale.x,
        normal.y * inv_scale.y,
        normal.z * inv_scale.z,
    );
    let n_world = orientation.rotate_vec(n_scaled).to_normalized();

    Vertex {
        position: [world.x, world.y, world.z],
        normal: [n_world.x, n_world.y, n_world.z],
        color,
    }
}

fn append_mesh_triangles(
    out: &mut Vec<Vertex>,
    mesh: &RenderMesh,
    position: Vec3,
    orientation: lin_alg::f32::Quaternion,
    scale: Vec3,
    color: [f32; 4],
    max_vertices: usize,
) -> bool {
    let inv_scale = inverse_scale(scale);

    for tri in mesh.indices.chunks_exact(3) {
        if out.len().saturating_add(3) > max_vertices {
            return false;
        }

        for &idx in tri {
            let Some(src) = mesh.vertices.get(idx) else {
                return false;
            };

            out.push(transform_vertex(
                Vec3::new(src.position[0], src.position[1], src.position[2]),
                Vec3::new(src.normal[0], src.normal[1], src.normal[2]),
                position,
                orientation,
                scale,
                inv_scale,
                color,
            ));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::molecule::{Atom, Element};
    use crate::viewer::{default_color_fn, MoleculeViewer};
    use lin_alg::f32::Vec3;

    fn molecule_with(atom_count: usize) -> Molecule {
        Molecule::from_atoms_bonds(
            (0..atom_count)
                .map(|i| Atom {
                    position: Vec3::new(i as f32 * 0.15, 0.0, 0.0),
                    element: Element::new("C"),
                    id: i,
                    meta: None,
                })
                .collect(),
            Vec::new(),
        )
    }

    /// Build the cache key the way `render_frame_with_state` does, from a
    /// viewer's current state.
    fn key_for(renderer: &OffscreenRenderer, viewer: &MoleculeViewer) -> GeometryCacheKey {
        let frame = RenderFrameState::new(
            viewer.molecule.as_ref(),
            [0.0; 16],
            None,
            1.0,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
            default_color_fn,
            None,
            RenderStyle::BallStick,
            16,
            false,
            viewer.molecule_opacity,
        )
        .with_geometry_revision(viewer.revision())
        .with_atom_attrs(
            viewer.atom_radii.as_deref(),
            viewer.atom_colors.as_deref(),
        );

        renderer.build_geometry_cache_key(&frame)
    }

    #[test]
    fn an_unchanged_viewer_keeps_the_same_key() {
        let renderer = OffscreenRenderer::new();
        let mut viewer = MoleculeViewer::new();
        viewer.set_molecule(molecule_with(3));

        assert_eq!(key_for(&renderer, &viewer), key_for(&renderer, &viewer));
    }

    /// The regression this whole key exists for: `set_molecule` writes into the
    /// viewer's existing `Option<Molecule>` slot, so a pointer-based key saw two
    /// different molecules as identical and left the first one on screen.
    #[test]
    fn replacing_the_molecule_invalidates_the_key() {
        let renderer = OffscreenRenderer::new();
        let mut viewer = MoleculeViewer::new();

        viewer.set_molecule(molecule_with(3));
        let first = key_for(&renderer, &viewer);

        viewer.set_molecule(molecule_with(5));
        let second = key_for(&renderer, &viewer);
        assert_ne!(first, second);

        // Even an identical replacement must invalidate: the caller asked for a
        // new molecule, and the renderer cannot cheaply prove it is the same.
        viewer.set_molecule(molecule_with(5));
        assert_ne!(second, key_for(&renderer, &viewer));
    }

    #[test]
    fn every_geometry_input_invalidates_the_key() {
        let renderer = OffscreenRenderer::new();
        let mut viewer = MoleculeViewer::new();
        viewer.set_molecule(molecule_with(3));

        let mut previous = key_for(&renderer, &viewer);
        let mut assert_changed = |viewer: &MoleculeViewer, what: &str| {
            let next = key_for(&renderer, viewer);
            assert_ne!(previous, next, "{what} should invalidate the geometry cache");
            previous = next;
        };

        viewer
            .update_positions(&[
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(2.0, 0.0, 0.0),
                Vec3::new(3.0, 0.0, 0.0),
            ])
            .expect("matching atom count");
        assert_changed(&viewer, "moving atoms");

        viewer.set_molecule_opacity(0.5);
        assert_changed(&viewer, "changing opacity");

        viewer.set_atom_radii(Some(vec![0.1, 0.2, 0.3]));
        assert_changed(&viewer, "setting per-atom radii");

        viewer.set_atom_radii(None);
        assert_changed(&viewer, "clearing per-atom radii");

        viewer.set_atom_colors(Some(vec![[1.0, 0.0, 0.0, 1.0]; 3]));
        assert_changed(&viewer, "setting per-atom colors");

        viewer.set_color_fn(|_, _| (0.0, 1.0, 0.0, 1.0));
        assert_changed(&viewer, "changing the color function");

        viewer.mark_changed();
        assert_changed(&viewer, "an explicit mark_changed");
    }

    /// Mesh resolution is tracked separately because the LOD worker can change
    /// it at the top of a render, after the caller assembled the frame.
    #[test]
    fn mesh_resolution_and_style_invalidate_the_key() {
        let mut renderer = OffscreenRenderer::new();
        let mut viewer = MoleculeViewer::new();
        viewer.set_molecule(molecule_with(3));

        let base = key_for(&renderer, &viewer);

        renderer.set_mesh_resolution(24);
        assert_ne!(base, key_for(&renderer, &viewer));

        renderer.set_mesh_resolution(16);
        renderer.set_render_style(RenderStyle::Wireframe);
        assert_ne!(base, key_for(&renderer, &viewer));
    }

    /// Normals transform by the inverse transpose of the model matrix, so a
    /// non-uniform scale must scale the normal by the *reciprocal*. Overlay
    /// geometry used to skip that correction while the molecule path applied it,
    /// and `add_cylinder` always sets a `(radius, length, radius)` scale -- so
    /// every overlay cylinder was lit as though it were unscaled.
    #[test]
    fn non_uniform_scale_corrects_the_normal() {
        use crate::scene_types::{Entity, Mesh, Vertex as SceneVertex};
        use lin_alg::f32::Quaternion;

        // A normal at 45 degrees in the XY plane, so stretching Y has to bend it.
        let diagonal = Vec3::new(1.0, 1.0, 0.0).to_normalized();
        let mut scene = Scene::default();
        scene.meshes.push(Mesh {
            vertices: vec![
                SceneVertex::new([0.0, 0.0, 0.0], diagonal),
                SceneVertex::new([1.0, 0.0, 0.0], diagonal),
                SceneVertex::new([0.0, 1.0, 0.0], diagonal),
            ],
            indices: vec![0, 1, 2],
        });

        let mut entity = Entity::new(
            0,
            Vec3::new(0.0, 0.0, 0.0),
            Quaternion::new_identity(),
            1.0,
            (1.0, 1.0, 1.0, 1.0),
            1.0,
        );
        entity.scale_partial = Some(Vec3::new(1.0, 4.0, 1.0));
        scene.entities.push(entity);

        let mut out = Vec::new();
        append_scene_triangles(&scene, &mut out, MAX_RENDER_VERTICES);
        assert_eq!(out.len(), 3);

        let expected = Vec3::new(1.0, 1.0 / 4.0, 0.0).to_normalized();
        for vertex in &out {
            for (got, want) in vertex.normal.iter().zip([expected.x, expected.y, expected.z]) {
                assert!(
                    (got - want).abs() < 1e-5,
                    "normal {:?} should be {expected:?}",
                    vertex.normal
                );
            }
        }
    }

    #[test]
    fn scene_geometry_respects_the_vertex_budget() {
        use crate::scene_types::{Entity, Mesh, Vertex as SceneVertex};
        use lin_alg::f32::Quaternion;

        let up = Vec3::new(0.0, 1.0, 0.0);
        let mut scene = Scene::default();
        scene.meshes.push(Mesh {
            vertices: vec![
                SceneVertex::new([0.0, 0.0, 0.0], up),
                SceneVertex::new([1.0, 0.0, 0.0], up),
                SceneVertex::new([0.0, 1.0, 0.0], up),
            ],
            indices: vec![0, 1, 2],
        });
        for _ in 0..10 {
            scene.entities.push(Entity::new(
                0,
                Vec3::new(0.0, 0.0, 0.0),
                Quaternion::new_identity(),
                1.0,
                (1.0, 1.0, 1.0, 1.0),
                1.0,
            ));
        }

        let mut out = Vec::new();
        append_scene_triangles(&scene, &mut out, 12);
        assert_eq!(out.len(), 12, "should stop at the budget, on a triangle boundary");
    }

    /// The scratch is reused across frames, so `clear` has to keep the storage
    /// it grew -- otherwise the retained batch buys nothing.
    #[test]
    fn clearing_a_scene_keeps_its_capacity() {
        let mut scene = Scene::default();
        let first = scene.unit_cylinder_mesh(10);
        let again = scene.unit_cylinder_mesh(10);
        assert_eq!(first, again, "a repeated request reuses the generated mesh");
        assert_eq!(scene.meshes.len(), 1);

        // A different resolution is a different mesh.
        assert_ne!(first, scene.unit_sphere_mesh(6, 12));
        assert_eq!(scene.meshes.len(), 2);

        let capacity = scene.meshes.capacity();
        scene.clear();
        assert!(scene.meshes.is_empty());
        assert_eq!(scene.meshes.capacity(), capacity);

        // Indices were invalidated by the clear, so the memo must not hand back
        // a stale one.
        assert_eq!(scene.unit_cylinder_mesh(10), 0);
    }

    /// A frame assembled by hand, without `with_geometry_revision`, must never
    /// hit the cache -- the renderer has no way to know whether it is stale.
    #[test]
    fn a_frame_without_a_declared_revision_never_caches() {
        let renderer = OffscreenRenderer::new();
        let molecule = molecule_with(3);

        let build = || {
            renderer.build_geometry_cache_key(&RenderFrameState::new(
                Some(&molecule),
                [0.0; 16],
                None,
                1.0,
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(0.0, 0.0, 1.0),
                default_color_fn,
                None,
                RenderStyle::BallStick,
                16,
                false,
                1.0,
            ))
        };

        assert_ne!(build(), build());
    }
}
