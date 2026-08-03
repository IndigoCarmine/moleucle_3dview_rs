use egui_wgpu::wgpu;
use wgpu::util::DeviceExt;

use super::render_styles::circles::CircleInstance;
use super::{BondInstance, BondMeshVertex, CircleQuadVertex, RenderMesh, Uniforms, Vertex};
use super::DEFAULT_BOND_CYLINDER_SIDES;
use crate::periodic::MAX_PERIODIC_IMAGES;

/// One render pipeline in two variants that differ *only* in
/// `depth_write_enabled`. Depth *testing* (`LessEqual`) stays on in both.
///
/// Translucent draws pick `no_depth_write` so a faded fragment does not stamp
/// the depth buffer and cull whatever is behind it — otherwise lowering an
/// object's alpha changes its color but still hides everything further away.
///
/// Caveat: with depth writes off, translucent fragments are blended in *draw
/// order* rather than back-to-front, so overlapping translucent geometry is
/// order-dependent. This renderer does no depth sorting and no order-independent
/// transparency; that approximation is accepted.
pub(super) struct PipelineSet {
    depth_write: wgpu::RenderPipeline,
    no_depth_write: wgpu::RenderPipeline,
}

impl PipelineSet {
    pub(super) fn get(&self, depth_write: bool) -> &wgpu::RenderPipeline {
        if depth_write {
            &self.depth_write
        } else {
            &self.no_depth_write
        }
    }
}

/// Build both variants of a pipeline from a closure that takes the label and
/// the `depth_write` flag to bake in.
fn pipeline_set(
    label: &str,
    mut build: impl FnMut(&str, bool) -> wgpu::RenderPipeline,
) -> PipelineSet {
    PipelineSet {
        depth_write: build(label, true),
        no_depth_write: build(&format!("{label}-no-depth-write"), false),
    }
}

/// Depth state shared by every pipeline: always depth-tested with `LessEqual`,
/// writing depth only when `depth_write` is set.
fn depth_stencil_state(depth_write: bool) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: wgpu::TextureFormat::Depth24Plus,
        depth_write_enabled: Some(depth_write),
        depth_compare: Some(wgpu::CompareFunction::LessEqual),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// One periodic image's translation. Padded to 16 bytes so it lines up with the
/// std140 rules the WGSL side is compiled against.
#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct ImageUniform {
    pub(super) translation: [f32; 3],
    pub(super) _pad: f32,
}

pub(super) struct GpuResources {
    pub(super) pipeline: PipelineSet,
    pub(super) additional_pipeline: PipelineSet,
    pub(super) wire_pipeline: PipelineSet,
    pub(super) circles_pipeline: PipelineSet,
    pub(super) bond_pipeline: PipelineSet,
    pub(super) uniform_buffer: wgpu::Buffer,
    pub(super) uniform_bind_group: wgpu::BindGroup,
    /// Per-periodic-image translations, one aligned slot each, read through a
    /// dynamic offset. See [`super::OffscreenRenderer::upload_image_translations`].
    pub(super) image_buffer: wgpu::Buffer,
    pub(super) image_bind_group: wgpu::BindGroup,
    /// Distance between image slots, i.e. the device's minimum uniform binding
    /// alignment.
    pub(super) image_stride: u32,
    pub(super) vertex_buffer: Option<wgpu::Buffer>,
    pub(super) vertex_count: u32,
    /// Allocated byte capacity of `vertex_buffer`, so equal-size mesh rebuilds
    /// (e.g. trajectory frames of a small molecule) reuse it via `write_buffer`.
    pub(super) vertex_capacity: usize,
    pub(super) circles_quad_buffer: wgpu::Buffer,
    pub(super) circles_instance_buffer: Option<wgpu::Buffer>,
    pub(super) circles_instance_count: u32,
    /// Allocated byte capacity of `circles_instance_buffer`, so trajectory
    /// frames of equal size reuse it via `write_buffer` instead of reallocating.
    pub(super) circles_instance_capacity: usize,
    /// Unit cylinder mesh (non-indexed triangle list) shared by every bond.
    pub(super) bond_mesh_buffer: wgpu::Buffer,
    pub(super) bond_mesh_vertex_count: u32,
    pub(super) bond_instance_buffer: Option<wgpu::Buffer>,
    pub(super) bond_instance_count: u32,
    pub(super) bond_instance_capacity: usize,
}

/// Upload `data` into a persistent instance buffer, reusing the existing
/// allocation via `queue.write_buffer` when it still fits. Returns the new
/// instance count. Reusing the buffer keeps trajectory playback (fixed
/// topology, only positions change) from reallocating a large GPU buffer every
/// frame.
pub(super) fn upload_instances<T: bytemuck::Pod>(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    buffer: &mut Option<wgpu::Buffer>,
    capacity_bytes: &mut usize,
    label: &str,
    data: &[T],
) -> u32 {
    if data.is_empty() {
        return 0;
    }
    let bytes: &[u8] = bytemuck::cast_slice(data);

    let fits = buffer.is_some() && bytes.len() <= *capacity_bytes;
    if fits {
        queue.write_buffer(buffer.as_ref().unwrap(), 0, bytes);
    } else {
        *buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytes,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        }));
        *capacity_bytes = bytes.len();
    }
    data.len() as u32
}

pub(super) fn create_gpu_resources(device: &wgpu::Device) -> GpuResources {
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

    // Periodic images are drawn by replaying the same geometry with a different
    // translation. Rather than duplicating vertices (27 copies of a large
    // molecule is not affordable) or baking an image index into the instance
    // data (which would collide with the per-atom instance attributes the
    // impostor pipelines already use), each image is one aligned slot in a
    // uniform buffer selected by a dynamic offset. Cost per image: 16 bytes and
    // one `set_bind_group`.
    let image_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("offscreen-image-layout"),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: wgpu::BufferSize::new(
                    std::mem::size_of::<ImageUniform>() as u64
                ),
            },
            count: None,
        }],
    });

    let image_stride = device
        .limits()
        .min_uniform_buffer_offset_alignment
        .max(std::mem::size_of::<ImageUniform>() as u32);
    let image_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("offscreen-image-buffer"),
        size: image_stride as u64 * MAX_PERIODIC_IMAGES as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let image_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("offscreen-image-bind-group"),
        layout: &image_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &image_buffer,
                offset: 0,
                size: wgpu::BufferSize::new(std::mem::size_of::<ImageUniform>() as u64),
            }),
        }],
    });

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("offscreen-shader"),
        source: wgpu::ShaderSource::Wgsl(MESH_SHADER.into()),
    });

    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("offscreen-layout"),
        bind_group_layouts: &[Some(&uniform_layout), Some(&image_layout)],
        immediate_size: 0,
    });

    let pipeline = pipeline_set("offscreen-pipeline", |label, depth_write| {
        create_triangle_pipeline(device, &layout, &shader, label, depth_write)
    });
    let additional_pipeline =
        pipeline_set("offscreen-additional-pipeline", |label, depth_write| {
            create_triangle_pipeline(device, &layout, &shader, label, depth_write)
        });

    let wire_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("offscreen-wire-shader"),
        source: wgpu::ShaderSource::Wgsl(WIRE_SHADER.into()),
    });

    let wire_pipeline = pipeline_set("offscreen-wire-pipeline", |label, depth_write| {
        create_wire_pipeline(device, &layout, &wire_shader, label, depth_write)
    });

    let circles_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("offscreen-circles-shader"),
        source: wgpu::ShaderSource::Wgsl(CIRCLES_SHADER.into()),
    });
    let circles_pipeline = pipeline_set("offscreen-circles-pipeline", |label, depth_write| {
        create_circles_pipeline(device, &layout, &circles_shader, label, depth_write)
    });
    let circles_quad_vertices = [
        CircleQuadVertex { corner: [-1.0, -1.0] },
        CircleQuadVertex { corner: [1.0, -1.0] },
        CircleQuadVertex { corner: [1.0, 1.0] },
        CircleQuadVertex { corner: [-1.0, -1.0] },
        CircleQuadVertex { corner: [1.0, 1.0] },
        CircleQuadVertex { corner: [-1.0, 1.0] },
    ];
    let circles_quad_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offscreen-circles-quad-buffer"),
        contents: bytemuck::cast_slice(&circles_quad_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let bond_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("offscreen-bond-shader"),
        source: wgpu::ShaderSource::Wgsl(BOND_SHADER.into()),
    });
    let bond_pipeline = pipeline_set("offscreen-bond-pipeline", |label, depth_write| {
        create_bond_pipeline(device, &layout, &bond_shader, label, depth_write)
    });
    // Expand the indexed unit cylinder into a non-indexed triangle list, the
    // shared geometry every bond instance is drawn with.
    let cylinder = RenderMesh::new_cylinder_open_ended(1.0, 1.0, DEFAULT_BOND_CYLINDER_SIDES);
    let bond_mesh_vertices: Vec<BondMeshVertex> = cylinder
        .indices
        .iter()
        .map(|&idx| {
            let v = cylinder.vertices[idx];
            BondMeshVertex {
                position: v.position,
                normal: v.normal,
            }
        })
        .collect();
    let bond_mesh_vertex_count = bond_mesh_vertices.len() as u32;
    let bond_mesh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("offscreen-bond-mesh-buffer"),
        contents: bytemuck::cast_slice(&bond_mesh_vertices),
        usage: wgpu::BufferUsages::VERTEX,
    });

    GpuResources {
        pipeline,
        additional_pipeline,
        wire_pipeline,
        circles_pipeline,
        bond_pipeline,
        uniform_buffer,
        uniform_bind_group,
        image_buffer,
        image_bind_group,
        image_stride,
        vertex_buffer: None,
        vertex_count: 0,
        vertex_capacity: 0,
        circles_quad_buffer,
        circles_instance_buffer: None,
        circles_instance_count: 0,
        circles_instance_capacity: 0,
        bond_mesh_buffer,
        bond_mesh_vertex_count,
        bond_instance_buffer: None,
        bond_instance_count: 0,
        bond_instance_capacity: 0,
    }
}

fn create_triangle_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    depth_write: bool,
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
                        format: wgpu::VertexFormat::Float32x4,
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
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
        depth_stencil: Some(depth_stencil_state(depth_write)),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_wire_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    depth_write: bool,
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
                        format: wgpu::VertexFormat::Float32x4,
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
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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
        depth_stencil: Some(depth_stencil_state(depth_write)),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_circles_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    depth_write: bool,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
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
                            format: wgpu::VertexFormat::Float32x4,
                            offset: std::mem::size_of::<[f32; 4]>() as u64,
                            shader_location: 3,
                        },
                    ],
                },
            ],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
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
        depth_stencil: Some(depth_stencil_state(depth_write)),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

fn create_bond_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    label: &str,
    depth_write: bool,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[
                // Per-vertex unit cylinder geometry.
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<BondMeshVertex>() as u64,
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
                    ],
                },
                // Per-bond instance data: mid, radius, axis, length, color.
                wgpu::VertexBufferLayout {
                    array_stride: std::mem::size_of::<BondInstance>() as u64,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: 0,
                            shader_location: 2,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: std::mem::size_of::<[f32; 3]>() as u64,
                            shader_location: 3,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x3,
                            offset: std::mem::size_of::<[f32; 4]>() as u64,
                            shader_location: 4,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32,
                            offset: std::mem::size_of::<[f32; 7]>() as u64,
                            shader_location: 5,
                        },
                        wgpu::VertexAttribute {
                            format: wgpu::VertexFormat::Float32x4,
                            offset: std::mem::size_of::<[f32; 8]>() as u64,
                            shader_location: 6,
                        },
                    ],
                },
            ],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
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
            // Cylinder is open-ended; don't cull so it's visible from inside too.
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(depth_stencil_state(depth_write)),
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

const MESH_SHADER: &str = r#"
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
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

// Translation carrying the primary cell onto the periodic image being drawn.
// Zero for the primary cell itself. Selected per draw by a dynamic offset, so
// every image reuses the same geometry buffers.
struct ImageUniform {
    translation: vec3<f32>,
};

@group(1) @binding(0)
var<uniform> image: ImageUniform;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
) -> VSOut {
    var out: VSOut;
    let clip = uniforms.view_proj * vec4<f32>(position + image.translation, 1.0);
    out.position = clip;
    out.color = color;
    out.normal = normal;
    out.depth01 = clip.z / clip.w * 0.5 + 0.5;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let base = in.color.rgb;
    let n = normalize(in.normal);
    // Camera-relative key light — see the note in CIRCLES_SHADER: a world-fixed
    // direction made brightness depend on which way the camera happened to face.
    let l = normalize(
        uniforms.camera_right.xyz * 0.4
        + uniforms.camera_up.xyz * 0.6
        - uniforms.camera_forward.xyz
    );
    let diffuse = max(dot(n, l), 0.0);
    let ambient = 0.22;
    let standard_lit = base * (ambient + 0.78 * diffuse);

    // For deeper fragments, blend toward a sphere-like radial tone.
    let depth01 = clamp(in.depth01, 0.0, 1.0);
    let deep_factor = smoothstep(0.60, 0.98, depth01);
    let radial = clamp(n.z * 0.5 + 0.5, 0.0, 1.0);

    let core = base * 1.05 + vec3<f32>(0.10, 0.10, 0.10);
    let mid = base;
    let edge = base * 0.28;
    let center_mix = smoothstep(0.55, 1.0, radial);
    let radial_mix = smoothstep(0.05, 0.85, radial);
    var sphere_tone = mix(edge, mix(mid, core, center_mix), radial_mix);

    // Darker rim to emulate the sample's white->base->black feel.
    let rim = mix(0.72, 1.0, smoothstep(0.0, 0.55, radial));
    sphere_tone = sphere_tone * rim;

    let lit = mix(standard_lit, sphere_tone, deep_factor);
    return vec4<f32>(lit, in.color.a);
}
"#;

const WIRE_SHADER: &str = r#"
struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
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

// Translation carrying the primary cell onto the periodic image being drawn.
// Zero for the primary cell itself. Selected per draw by a dynamic offset, so
// every image reuses the same geometry buffers.
struct ImageUniform {
    translation: vec3<f32>,
};

@group(1) @binding(0)
var<uniform> image: ImageUniform;

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) _normal: vec3<f32>,
    @location(2) color: vec4<f32>,
) -> VSOut {
    var out: VSOut;
    out.position = uniforms.view_proj * vec4<f32>(position + image.translation, 1.0);
    out.color = color;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

const CIRCLES_SHADER: &str = r#"
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

// Translation carrying the primary cell onto the periodic image being drawn.
// Zero for the primary cell itself. Selected per draw by a dynamic offset, so
// every image reuses the same geometry buffers.
struct ImageUniform {
    translation: vec3<f32>,
};

@group(1) @binding(0)
var<uniform> image: ImageUniform;

struct VSOut {
    @builtin(position) position: vec4<f32>,

    // sphere center in world/view space
    @location(0) center: vec3<f32>,

    @location(1) radius: f32,
    @location(2) color: vec4<f32>,
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
    @location(3) color: vec4<f32>,
) -> VSOut {
    var out: VSOut;

    let center_ws = center + image.translation;

    let clip_center =
        uniforms.view_proj * vec4<f32>(center_ws, 1.0);

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

    out.center = center_ws;
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

    // Key light fixed relative to the camera rather than the world. A
    // world-fixed direction leaves the camera-facing hemisphere unlit over half
    // of all orbits, collapsing everything to `ambient` — mean brightness
    // measured 3.4x apart between camera angles, and the default view happened
    // to be one of the dark ones. Offset up and to the right of the viewer so
    // spheres still shade across their face instead of reading as flat discs.
    let light_dir = normalize(
        uniforms.camera_right.xyz * 0.4
        + uniforms.camera_up.xyz * 0.6
        - uniforms.camera_forward.xyz
    );

    let diffuse =
        max(dot(world_normal, light_dir), 0.0);

    let ambient = 0.15;

    let lit =
        in.color.rgb * (ambient + diffuse * 0.85);

    out.color = vec4<f32>(lit, in.color.a);

    return out;
}
"#;

const BOND_SHADER: &str = r#"
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

// Translation carrying the primary cell onto the periodic image being drawn.
// Zero for the primary cell itself. Selected per draw by a dynamic offset, so
// every image reuses the same geometry buffers.
struct ImageUniform {
    translation: vec3<f32>,
};

@group(1) @binding(0)
var<uniform> image: ImageUniform;

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) normal: vec3<f32>,
};

@vertex
fn vs_main(
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) mid: vec3<f32>,
    @location(3) radius: f32,
    @location(4) axis: vec3<f32>,
    @location(5) length: f32,
    @location(6) color: vec4<f32>,
) -> VSOut {
    // Orthonormal basis perpendicular to the (unit) cylinder axis.
    var up = vec3<f32>(0.0, 1.0, 0.0);
    if (abs(axis.y) > 0.99) {
        up = vec3<f32>(1.0, 0.0, 0.0);
    }
    let right = normalize(cross(up, axis));
    let forward = cross(axis, right);

    // Unit cylinder: position.xz on the circle, position.y in [-0.5, 0.5].
    let world_pos =
        mid
        + right * (position.x * radius)
        + axis * (position.y * length)
        + forward * (position.z * radius);

    // Radial normal lies in the right/forward plane.
    let world_normal =
        normalize(right * normal.x + forward * normal.z);

    var out: VSOut;
    out.position = uniforms.view_proj * vec4<f32>(world_pos + image.translation, 1.0);
    out.color = color;
    out.normal = world_normal;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    // Camera-relative key light, matching the atom shaders so bonds and the
    // spheres they join are lit from the same side.
    let l = normalize(
        uniforms.camera_right.xyz * 0.4
        + uniforms.camera_up.xyz * 0.6
        - uniforms.camera_forward.xyz
    );
    let diffuse = max(dot(n, l), 0.0);
    let ambient = 0.25;
    return vec4<f32>(in.color.rgb * (ambient + 0.75 * diffuse), in.color.a);
}
"#;

#[cfg(test)]
mod shader_tests {
    use super::*;

    fn validate(name: &str, src: &str) {
        let module = naga::front::wgsl::parse_str(src)
            .unwrap_or_else(|e| panic!("{name} failed to parse:\n{}", e.emit_to_string(src)));
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .unwrap_or_else(|e| panic!("{name} failed validation: {e:?}"));
    }

    #[test]
    fn all_wgsl_shaders_compile() {
        validate("MESH_SHADER", MESH_SHADER);
        validate("WIRE_SHADER", WIRE_SHADER);
        validate("CIRCLES_SHADER", CIRCLES_SHADER);
        validate("BOND_SHADER", BOND_SHADER);
    }
}
