use lin_alg::f32::{Quaternion, Vec3};

#[derive(Clone, Copy, Debug)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: Vec3,
}

impl Vertex {
    pub fn new(position: [f32; 3], normal: Vec3) -> Self {
        Self { position, normal }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<usize>,
}

impl Mesh {
    pub fn new_sphere(radius: f32, subdivisions: u32) -> Self {
        let lat_segments = 8 + subdivisions as usize * 6;
        let lon_segments = 12 + subdivisions as usize * 8;
        Self::new_sphere_uv(radius, lat_segments, lon_segments)
    }

    pub fn new_sphere_uv(radius: f32, lat_segments: usize, lon_segments: usize) -> Self {
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
                vertices.push(Vertex::new([x, y, z], n));
            }
        }

        let row = lon_segments + 1;
        for lat in 0..lat_segments {
            for lon in 0..lon_segments {
                let i0 = lat * row + lon;
                let i1 = i0 + 1;
                let i2 = i0 + row;
                let i3 = i2 + 1;
                indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
            }
        }

        Self { vertices, indices }
    }

    pub fn new_cylinder(len: f32, radius: f32, sides: usize) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let half = len * 0.5;

        for i in 0..sides {
            let t = i as f32 / sides as f32;
            let angle = t * std::f32::consts::TAU;
            let x = radius * angle.cos();
            let z = radius * angle.sin();
            let n = Vec3::new(x, 0.0, z).to_normalized();

            vertices.push(Vertex::new([x, half, z], n));
            vertices.push(Vertex::new([x, -half, z], n));
        }

        for i in 0..sides {
            let next = (i + 1) % sides;
            let top0 = i * 2;
            let bot0 = top0 + 1;
            let top1 = next * 2;
            let bot1 = top1 + 1;
            indices.extend_from_slice(&[top0, bot0, top1, top1, bot0, bot1]);
        }

        let top_center = vertices.len();
        vertices.push(Vertex::new([0.0, half, 0.0], Vec3::new(0.0, 1.0, 0.0)));
        let bottom_center = vertices.len();
        vertices.push(Vertex::new([0.0, -half, 0.0], Vec3::new(0.0, -1.0, 0.0)));

        for i in 0..sides {
            let next = (i + 1) % sides;
            let top0 = i * 2;
            let top1 = next * 2;
            let bot0 = top0 + 1;
            let bot1 = top1 + 1;
            indices.extend_from_slice(&[top_center, top1, top0, bottom_center, bot0, bot1]);
        }

        Self { vertices, indices }
    }
}

#[derive(Clone, Debug)]
pub struct Entity {
    pub mesh: usize,
    pub position: Vec3,
    pub orientation: Quaternion,
    pub scale: f32,
    pub scale_partial: Option<Vec3>,
    pub color: (f32, f32, f32),
    pub opacity: f32,
    pub shinyness: f32,
}

impl Entity {
    pub fn new(
        mesh: usize,
        position: Vec3,
        orientation: Quaternion,
        scale: f32,
        color: (f32, f32, f32),
        shinyness: f32,
    ) -> Self {
        Self {
            mesh,
            position,
            orientation,
            scale,
            scale_partial: None,
            color,
            opacity: 1.0,
            shinyness,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SceneCamera {
    pub position: Vec3,
    pub orientation: Quaternion,
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,
    pub aspect: f32,
}

impl Default for SceneCamera {
    fn default() -> Self {
        Self {
            position: Vec3::new_zero(),
            orientation: Quaternion::new_identity(),
            fov_y: 45.0f32.to_radians(),
            near: 0.1,
            far: 1000.0,
            aspect: 1.0,
        }
    }
}

impl SceneCamera {
    pub fn update_proj_mat(&mut self) {}
}

#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub entities: Vec<Entity>,
}

impl Scene {
    pub fn to_vertices(&self) -> Vec<Vertex> {
        let mut vertices = Vec::new();
        for entity in &self.entities {
            let mesh = &self.meshes[entity.mesh];
            for idx in &mesh.indices {
                let v = &mesh.vertices[*idx];
                let pos = Vec3::from(v.position) * entity.scale;
                let final_pos = pos + entity.position;
                vertices.push(Vertex::new(final_pos.to_arr(), v.normal));
            }
        }
        vertices
    }
}
