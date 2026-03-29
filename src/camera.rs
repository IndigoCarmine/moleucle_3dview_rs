
use nalgebra::{
    Isometry3, Matrix4,Perspective3, Point3, Unit, UnitQuaternion, Vector2, Vector3, Vector4
};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProjectionType {
    Perspective,
    Orthographic,
}

pub trait Camera {
    fn view_matrix(&self) -> Matrix4<f32>;
    fn projection_matrix(&self) -> Matrix4<f32>;
    fn view_projection(&self) -> Matrix4<f32> {
        self.projection_matrix() * self.view_matrix()
    }
    fn camera_rotation(&self) -> UnitQuaternion<f32>;
    fn position(&self) -> Point3<f32>;
    fn target(&self) -> Point3<f32>;
    fn up(&self) -> Vector3<f32>;

    fn set_aspect(&mut self, aspect: f32);

    fn orbit(&mut self, delta_x: f32, delta_y: f32);
    fn pan(&mut self, delta: Vector2<f32>);
    fn dolly(&mut self, delta: f32);

    fn fov_y(&self) -> f32;
    fn near(&self) -> f32;
    fn far(&self) -> f32;

    // Optional helper to set look_at if possible, otherwise it might be specific implementation dependent
    fn look_at(&mut self, eye: Point3<f32>, target: Point3<f32>, up: Vector3<f32>);

    // Ray casting from screen coordinates to world coordinates
    // u, v: screen coordinates (pixels)
    // width, height: screen dimensions (pixels)
    // returns: (origin, direction)
    fn ray_from_screen(
        &self,
        u: f32,
        v: f32,
        width: f32,
        height: f32,
    ) -> (lin_alg::f32::Vec3, lin_alg::f32::Vec3) {


        let inv_vp = self
            .view_projection()
            .try_inverse()
            .unwrap_or_else(Matrix4::identity);

        // NDC
        let ndc_x = 1.0 - 2.0 * u / width;
        let ndc_y = 1.0 - 2.0 * v / height;

        // OpenGL depth
        let near = Vector4::new(ndc_x, ndc_y, -1.0, 1.0);
        let far  = Vector4::new(ndc_x, ndc_y,  1.0, 1.0);

        let world_near = inv_vp * near;
        let world_far  = inv_vp * far;

        let p_near = world_near.xyz() / world_near.w;
        let p_far  = world_far.xyz()  / world_far.w;

        let dir = (p_far - p_near).normalize();

        (
            lin_alg::f32::Vec3::new(p_near.x, p_near.y, p_near.z),
            lin_alg::f32::Vec3::new(dir.x, dir.y, dir.z),
        )
    }
}


// =========================================================================
// Orbital Camera
// =========================================================================

pub struct OrbitalCamera {
    pub center: Point3<f32>,
    pub rotation: UnitQuaternion<f32>,
    pub radius: f32,

    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    
    // For debug visualization: last orbit rotation axis
    pub last_orbit_axis: Vector3<f32>,
}

impl Default for OrbitalCamera {
    fn default() -> Self {
        Self {
            center: Point3::origin(),
            rotation: UnitQuaternion::identity(),
            radius: 10.0,
            fov_y: 45.0f32.to_radians(),
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
            last_orbit_axis: Vector3::z(), // Default: Z-axis
        }
    }
}

impl Camera for OrbitalCamera {
    fn view_matrix(&self) -> Matrix4<f32> {
        let eye = self.position();
        let target = self.target();
        let up = self.up();

        Isometry3::look_at_rh(&eye, &target, &up).to_homogeneous()
    }
    fn camera_rotation(&self) -> UnitQuaternion<f32> {
        self.rotation
    }

    fn projection_matrix(&self) -> Matrix4<f32> {
        Perspective3::new(self.aspect, self.fov_y, self.near, self.far).to_homogeneous()
    }

    fn position(&self) -> Point3<f32> {
        self.center - self.rotation * Vector3::new(0.0, 0.0, self.radius)
    }

    fn target(&self) -> Point3<f32> {
        self.center
    }

    fn up(&self) -> Vector3<f32> {
        self.rotation * Vector3::y()
    }

    fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        // Calculate rotation axis in world space
        let rot_axis = self.rotation * Vector3::new(delta_y, delta_x, 0.0).normalize();
        // let rot_axis = self.rotation * Vector3::new(1.0,0.1,0.0);
        let len = (delta_x * delta_x + delta_y * delta_y).sqrt();
        
        // Store the axis for debug visualization
        self.last_orbit_axis = rot_axis;
        
        self.rotation = UnitQuaternion::from_axis_angle(
            &Unit::new_normalize(rot_axis), len ) * self.rotation;
    }

    fn pan(&mut self, delta: Vector2<f32>) {
        // Pan moves the center.
        // Move along local Right and Up.
        let scale = self.radius * 0.1; // Adjust pan speed based on distance
        let right = self.rotation * Vector3::x() * -scale;
        let up = self.rotation * Vector3::y() * scale;

        self.center += right * delta.x + up * delta.y;
    }

    fn dolly(&mut self, delta: f32) {
        self.radius = (self.radius - delta).max(0.1);
    }

    fn fov_y(&self) -> f32 {
        self.fov_y
    }
    fn near(&self) -> f32 {
        self.near
    }
    fn far(&self) -> f32 {
        self.far
    }

    fn look_at(&mut self, eye: Point3<f32>, target: Point3<f32>, up: Vector3<f32>) {
        self.center = target;

        let dir = eye - target;
        self.radius = dir.magnitude();

        let iso = Isometry3::look_at_rh(&eye, &target, &up);
        self.rotation = iso.rotation.inverse();
    }

}
