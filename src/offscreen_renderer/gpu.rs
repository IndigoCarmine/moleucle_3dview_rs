use egui_wgpu::wgpu;
use wgpu::util::DeviceExt;

use super::render_styles::circles::CircleInstance;
use super::{BondInstance, BondMeshVertex, CircleQuadVertex, RenderMesh, Uniforms, Vertex};
use super::DEFAULT_BOND_CYLINDER_SIDES;

pub(super) struct GpuResources {
    pub(super) pipeline: wgpu::RenderPipeline,
    pub(super) additional_pipeline: wgpu::RenderPipeline,
    pub(super) wire_pipeline: wgpu::RenderPipeline,
    pub(super) circles_pipeline: wgpu::RenderPipeline,
    pub(super) bond_pipeline: wgpu::RenderPipeline,
    pub(super) uniform_buffer: wgpu::Buffer,
    pub(super) uniform_bind_group: wgpu::BindGroup,
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

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("offscreen-shader"),
        source: wgpu::ShaderSource::Wgsl(MESH_SHADER.into()),
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
        source: wgpu::ShaderSource::Wgsl(WIRE_SHADER.into()),
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

    let bond_pipeline = create_bond_pipeline(device, &layout);
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
        source: wgpu::ShaderSource::Wgsl(CIRCLES_SHADER.into()),
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

fn create_bond_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("offscreen-bond-shader"),
        source: wgpu::ShaderSource::Wgsl(BOND_SHADER.into()),
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("offscreen-bond-pipeline"),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: &shader,
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
                            format: wgpu::VertexFormat::Float32x3,
                            offset: std::mem::size_of::<[f32; 8]>() as u64,
                            shader_location: 6,
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
                blend: Some(wgpu::BlendState::REPLACE),
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

const MESH_SHADER: &str = r#"
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
"#;

const WIRE_SHADER: &str = r#"
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

struct VSOut {
    @builtin(position) position: vec4<f32>,
    @location(0) color: vec3<f32>,
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
    @location(6) color: vec3<f32>,
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
    out.position = uniforms.view_proj * vec4<f32>(world_pos, 1.0);
    out.color = color;
    out.normal = world_normal;
    return out;
}

@fragment
fn fs_main(in: VSOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let l = normalize(vec3<f32>(0.35, 0.75, 0.55));
    let diffuse = max(dot(n, l), 0.0);
    let ambient = 0.25;
    return vec4<f32>(in.color * (ambient + 0.75 * diffuse), 1.0);
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
