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
    /// RGBA color. The fourth component is the alpha channel used for blending
    /// (1.0 = opaque, 0.0 = fully transparent).
    pub color: (f32, f32, f32, f32),
    pub opacity: f32,
    pub shinyness: f32,
}

impl Entity {
    pub fn new(
        mesh: usize,
        position: Vec3,
        orientation: Quaternion,
        scale: f32,
        color: (f32, f32, f32, f32),
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

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SphereImpostorInstance {
    pub center: [f32; 3],
    pub radius: f32,
    /// RGBA color; the alpha component drives alpha blending in the shader.
    pub color: [f32; 4],
}

/// CPU-side geometry an [`crate::AdditionalRender`] contributes to a frame.
///
/// `meshes` holds the distinct shapes and `entities` places, scales and colours
/// them, so drawing a thousand identical spheres costs one mesh and a thousand
/// small entities. [`Scene::unit_sphere_mesh`] and [`Scene::unit_cylinder_mesh`]
/// exist to make that the easy path: they generate a unit primitive once per
/// scene and hand back the same index on every subsequent call.
#[derive(Clone, Debug, Default)]
pub struct Scene {
    pub meshes: Vec<Mesh>,
    pub entities: Vec<Entity>,
    pub sphere_impostors: Vec<SphereImpostorInstance>,
    /// `(lat, lon) -> meshes` index of the unit spheres generated so far.
    unit_spheres: Vec<((usize, usize), usize)>,
    /// `sides -> meshes` index of the unit cylinders generated so far.
    unit_cylinders: Vec<(usize, usize)>,
}

impl Scene {
    /// Index of a unit-radius sphere at this resolution, generating it on first
    /// use and reusing it afterwards.
    ///
    /// Overlays that draw many spheres previously built a fresh UV sphere per
    /// sphere per frame — hundreds of vertices allocated and thrown away for
    /// every highlighted atom, every frame.
    pub fn unit_sphere_mesh(&mut self, lat_segments: usize, lon_segments: usize) -> usize {
        let key = (lat_segments, lon_segments);
        if let Some((_, index)) = self.unit_spheres.iter().find(|(k, _)| *k == key) {
            return *index;
        }

        let index = self.meshes.len();
        self.meshes
            .push(Mesh::new_sphere_uv(1.0, lat_segments, lon_segments));
        self.unit_spheres.push((key, index));
        index
    }

    /// Index of a unit-length, unit-radius cylinder along +Y with this many
    /// sides, generating it on first use and reusing it afterwards.
    pub fn unit_cylinder_mesh(&mut self, sides: usize) -> usize {
        if let Some((_, index)) = self.unit_cylinders.iter().find(|(k, _)| *k == sides) {
            return *index;
        }

        let index = self.meshes.len();
        self.meshes.push(Mesh::new_cylinder(1.0, 1.0, sides));
        self.unit_cylinders.push((sides, index));
        index
    }

    /// Empty the scene while keeping every allocation, so the next frame refills
    /// it without touching the allocator. The memoized unit primitives are
    /// dropped along with `meshes`, since their indices would no longer be valid.
    pub fn clear(&mut self) {
        self.meshes.clear();
        self.entities.clear();
        self.sphere_impostors.clear();
        self.unit_spheres.clear();
        self.unit_cylinders.clear();
    }
}
