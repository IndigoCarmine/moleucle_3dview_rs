use crate::Molecule;
use egui::TextureId;
use egui_wgpu::wgpu;
use lin_alg::f32::{Quaternion, Vec3};
use std::collections::HashSet;
use wgpu::util::DeviceExt;

const DEFAULT_MESH_RESOLUTION: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderStyle {
    BallStick,
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
struct Uniforms {
    view_proj: [f32; 16],
}

struct GpuResources {
    pipeline: wgpu::RenderPipeline,
    wire_pipeline: wgpu::RenderPipeline,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    vertex_buffer: Option<wgpu::Buffer>,
    vertex_count: u32,
}

pub struct OffscreenRenderer {
    width: u32,
    height: u32,
    mesh_resolution: usize,
    render_style: RenderStyle,
    color_texture: Option<wgpu::Texture>,
    color_view: Option<wgpu::TextureView>,
    depth_texture: Option<wgpu::Texture>,
    depth_view: Option<wgpu::TextureView>,
    texture_id: Option<TextureId>,
    gpu: Option<GpuResources>,
    sphere_mesh: RenderMesh,
    cylinder_mesh: RenderMesh,
}

impl OffscreenRenderer {
    pub fn new() -> Self {
        Self::new_with_mesh_resolution(DEFAULT_MESH_RESOLUTION)
    }

    pub fn new_with_mesh_resolution(mesh_resolution: usize) -> Self {
        let mesh_resolution = mesh_resolution.max(3);
        let sphere_lat = mesh_resolution;
        let sphere_lon = mesh_resolution * 2;
        let cylinder_sides = mesh_resolution * 2;

        Self {
            width: 0,
            height: 0,
            mesh_resolution,
            render_style: RenderStyle::BallStick,
            color_texture: None,
            color_view: None,
            depth_texture: None,
            depth_view: None,
            texture_id: None,
            gpu: None,
            sphere_mesh: RenderMesh::new_sphere_uv(1.0, sphere_lat, sphere_lon),
            cylinder_mesh: RenderMesh::new_cylinder(1.0, 1.0, cylinder_sides),
        }
    }

    pub fn set_mesh_resolution(&mut self, mesh_resolution: usize) {
        let mesh_resolution = mesh_resolution.max(3);
        if self.mesh_resolution == mesh_resolution {
            return;
        }

        self.mesh_resolution = mesh_resolution;
        let sphere_lat = mesh_resolution;
        let sphere_lon = mesh_resolution * 2;
        let cylinder_sides = mesh_resolution * 2;
        self.sphere_mesh = RenderMesh::new_sphere_uv(1.0, sphere_lat, sphere_lon);
        self.cylinder_mesh = RenderMesh::new_cylinder(1.0, 1.0, cylinder_sides);
    }

    pub fn render_style(&self) -> RenderStyle {
        self.render_style
    }

    pub fn set_render_style(&mut self, render_style: RenderStyle) {
        self.render_style = render_style;
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

        let needs_rebuild = self.color_view.is_none() || self.width != width || self.height != height;
        if needs_rebuild {
            self.width = width;
            self.height = height;
            self.rebuild_targets(render_state);
        }

        Ok(())
    }

    pub fn render_frame(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        molecule: Option<&Molecule>,
        selected_atoms: &[usize],
        view_proj: [f32; 16],
    ) -> Result<(), String> {
        let color_view = self
            .color_view
            .as_ref()
            .ok_or_else(|| "Offscreen color view is not initialized".to_string())?;
        let depth_view = self
            .depth_view
            .as_ref()
            .ok_or_else(|| "Offscreen depth view is not initialized".to_string())?;

        let vertices = self.build_scene_vertices(molecule, selected_atoms);

        let gpu = self
            .gpu
            .as_mut()
            .ok_or_else(|| "Offscreen GPU resources are not initialized".to_string())?;

        let uniforms = Uniforms { view_proj };
        render_state
            .queue
            .write_buffer(&gpu.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        if vertices.is_empty() {
            gpu.vertex_buffer = None;
            gpu.vertex_count = 0;
        } else {
            gpu.vertex_count = vertices.len() as u32;
            gpu.vertex_buffer = Some(render_state.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("offscreen-vertex-buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                },
            ));
        }

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
                    view: color_view,
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
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            let pipeline = match self.render_style {
                RenderStyle::BallStick => &gpu.pipeline,
                RenderStyle::Wireframe => &gpu.wire_pipeline,
            };
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &gpu.uniform_bind_group, &[]);
            if let Some(vertex_buffer) = &gpu.vertex_buffer {
                pass.set_vertex_buffer(0, vertex_buffer.slice(..));
                pass.draw(0..gpu.vertex_count, 0..1);
            }
        }

        render_state.queue.submit(Some(encoder.finish()));
        Ok(())
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
        let depth_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
        self.color_view = Some(color_view);
        self.depth_texture = Some(depth_texture);
        self.depth_view = Some(depth_view);
    }

    fn build_scene_vertices(&self, molecule: Option<&Molecule>, selected_atoms: &[usize]) -> Vec<Vertex> {
        match self.render_style {
            RenderStyle::BallStick => self.build_ballstick_vertices(molecule, selected_atoms),
            RenderStyle::Wireframe => self.build_wireframe_vertices(molecule, selected_atoms),
        }
    }

    fn build_ballstick_vertices(&self, molecule: Option<&Molecule>, selected_atoms: &[usize]) -> Vec<Vertex> {
        let mut vertices = Vec::new();

        if let Some(mol) = molecule {
            let selected: HashSet<usize> = selected_atoms.iter().copied().collect();

            for bond in &mol.bonds {
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

                let base_radius = if bond_order <= 1 { 0.15 } else { 0.10 };
                for offset in line_offsets {
                    append_mesh_triangles(
                        &mut vertices,
                        &self.cylinder_mesh,
                        mid + lateral * offset,
                        orientation,
                        Vec3::new(base_radius, len, base_radius),
                        [0.55, 0.55, 0.55],
                    );
                }
            }

            for (idx, atom) in mol.atoms.iter().enumerate() {
                let pos = atom.position;
                let selected_this = selected.contains(&idx);
                let radius = if selected_this { 0.56 } else { 0.40 };
                let color = if selected_this {
                    [1.0, 0.35, 0.10]
                } else {
                    element_color(&atom.element)
                };

                append_mesh_triangles(
                    &mut vertices,
                    &self.sphere_mesh,
                    pos,
                    Quaternion::new_identity(),
                    Vec3::new(radius, radius, radius),
                    color,
                );
            }
        }

        // xyz axes as cylinders
        let axis_len = 2.0;
        let axis_radius = 0.05;
        append_mesh_triangles(
            &mut vertices,
            &self.cylinder_mesh,
            Vec3::new(axis_len * 0.5, 0.0, 0.0),
            Quaternion::from_axis_angle(Vec3::new(0.0, 0.0, 1.0), -std::f32::consts::FRAC_PI_2),
            Vec3::new(axis_radius, axis_len, axis_radius),
            [1.0, 0.0, 0.0],
        );
        append_mesh_triangles(
            &mut vertices,
            &self.cylinder_mesh,
            Vec3::new(0.0, axis_len * 0.5, 0.0),
            Quaternion::new_identity(),
            Vec3::new(axis_radius, axis_len, axis_radius),
            [0.0, 1.0, 0.0],
        );
        append_mesh_triangles(
            &mut vertices,
            &self.cylinder_mesh,
            Vec3::new(0.0, 0.0, axis_len * 0.5),
            Quaternion::from_axis_angle(Vec3::new(1.0, 0.0, 0.0), std::f32::consts::FRAC_PI_2),
            Vec3::new(axis_radius, axis_len, axis_radius),
            [0.0, 0.0, 1.0],
        );

        vertices
    }

    fn build_wireframe_vertices(&self, molecule: Option<&Molecule>, selected_atoms: &[usize]) -> Vec<Vertex> {
        let mut vertices = Vec::new();

        if let Some(mol) = molecule {
            let selected: HashSet<usize> = selected_atoms.iter().copied().collect();

            for bond in &mol.bonds {
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
                    append_line(&mut vertices, a + off, b + off, [0.70, 0.70, 0.72]);
                }
            }

            for (idx, atom) in mol.atoms.iter().enumerate() {
                let pos = atom.position;
                let selected_this = selected.contains(&idx);
                let span = if selected_this { 0.22 } else { 0.14 };
                let color = if selected_this {
                    [1.0, 0.35, 0.10]
                } else {
                    element_color(&atom.element)
                };

                append_line(
                    &mut vertices,
                    pos + Vec3::new(-span, 0.0, 0.0),
                    pos + Vec3::new(span, 0.0, 0.0),
                    color,
                );
                append_line(
                    &mut vertices,
                    pos + Vec3::new(0.0, -span, 0.0),
                    pos + Vec3::new(0.0, span, 0.0),
                    color,
                );
                append_line(
                    &mut vertices,
                    pos + Vec3::new(0.0, 0.0, -span),
                    pos + Vec3::new(0.0, 0.0, span),
                    color,
                );
            }
        }

        let axis_len = 2.0;
        append_line(
            &mut vertices,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(axis_len, 0.0, 0.0),
            [1.0, 0.0, 0.0],
        );
        append_line(
            &mut vertices,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, axis_len, 0.0),
            [0.0, 1.0, 0.0],
        );
        append_line(
            &mut vertices,
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, axis_len),
            [0.0, 0.0, 1.0],
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
            visibility: wgpu::ShaderStages::VERTEX,
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
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
            0.0, 0.0, 1.0, 0.0,
            0.0, 0.0, 0.0, 1.0,
        ],
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
};

struct Uniforms {
    view_proj: mat4x4<f32>,
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
    out.position = uniforms.view_proj * vec4<f32>(position, 1.0);
    out.color = color;
    out.normal = normal;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(vec3<f32>(0.35, 0.75, 0.55));
    let diffuse = max(dot(n, l), 0.0);
    let ambient = 0.25;
    let lit = in.color * (ambient + 0.75 * diffuse);
    return vec4<f32>(lit, 1.0);
}
"#
            .into(),
        ),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("offscreen-layout"),
        bind_group_layouts: &[&uniform_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("offscreen-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
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
            module: &shader,
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
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

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
            depth_write_enabled: true,
            depth_compare: wgpu::CompareFunction::LessEqual,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    GpuResources {
        pipeline,
        wire_pipeline,
        uniform_buffer,
        uniform_bind_group,
        vertex_buffer: None,
        vertex_count: 0,
    }
}

fn element_color(element: &str) -> [f32; 3] {
    match element {
        "C" => [0.12, 0.12, 0.12],
        "H" => [0.90, 0.90, 0.90],
        "O" => [0.95, 0.15, 0.15],
        "N" => [0.20, 0.30, 0.95],
        "S" => [0.95, 0.85, 0.20],
        "P" => [1.00, 0.55, 0.15],
        "CL" => [0.10, 0.85, 0.20],
        _ => [0.70, 0.70, 0.70],
    }
}

fn bond_line_offsets(order: usize) -> Vec<f32> {
    match order {
        0 | 1 => vec![0.0],
        2 => vec![-0.16, 0.16],
        3 => vec![-0.26, 0.0, 0.26],
        n => {
            let spacing = 0.14;
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
) {
    let inv_scale = Vec3::new(
        if scale.x.abs() > 1e-6 { 1.0 / scale.x } else { 0.0 },
        if scale.y.abs() > 1e-6 { 1.0 / scale.y } else { 0.0 },
        if scale.z.abs() > 1e-6 { 1.0 / scale.z } else { 0.0 },
    );

    for tri in mesh.indices.chunks(3) {
        if tri.len() < 3 {
            continue;
        }

        for &idx in tri {
            let Some(src) = mesh.vertices.get(idx) else {
                continue;
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
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

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

                indices.extend_from_slice(&[i0, i2, i1]);
                indices.extend_from_slice(&[i1, i2, i3]);
            }
        }

        Self { vertices, indices }
    }

    fn new_cylinder(len: f32, radius: f32, sides: usize) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
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

            indices.extend_from_slice(&[top0, bot0, top1]);
            indices.extend_from_slice(&[top1, bot0, bot1]);
        }

        let top_center = vertices.len();
        vertices.push(RenderVertex {
            position: [0.0, half, 0.0],
            normal: [0.0, 1.0, 0.0],
        });
        let bottom_center = vertices.len();
        vertices.push(RenderVertex {
            position: [0.0, -half, 0.0],
            normal: [0.0, -1.0, 0.0],
        });

        for i in 0..sides {
            let next = (i + 1) % sides;
            let top0 = i * 2;
            let top1 = next * 2;
            let bot0 = top0 + 1;
            let bot1 = top1 + 1;

            indices.extend_from_slice(&[top_center, top1, top0]);
            indices.extend_from_slice(&[bottom_center, bot0, bot1]);
        }

        Self { vertices, indices }
    }
}

fn append_line(out: &mut Vec<Vertex>, a: Vec3, b: Vec3, color: [f32; 3]) {
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
}