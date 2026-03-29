use lin_alg::f32::{Mat4, Quaternion, Vec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProjectionType {
    Perspective,
    Orthographic,
}

pub trait Camera {
    fn view_matrix(&self) -> Mat4;
    fn projection_matrix(&self) -> Mat4;
    fn view_projection(&self) -> Mat4 {
        self.projection_matrix() * self.view_matrix()
    }
    fn camera_rotation(&self) -> Quaternion;
    fn position(&self) -> Vec3;
    fn target(&self) -> Vec3;
    fn up(&self) -> Vec3;

    fn set_aspect(&mut self, aspect: f32);

    fn orbit(&mut self, delta_x: f32, delta_y: f32);
    fn pan(&mut self, delta: Vec2);
    fn dolly(&mut self, delta: f32);

    fn fov_y(&self) -> f32;
    fn near(&self) -> f32;
    fn far(&self) -> f32;

    // Optional helper to set look_at if possible, otherwise it might be specific implementation dependent
    fn look_at(&mut self, eye: Vec3, target: Vec3, up: Vec3);

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
        // Convert pixel coords to NDC, with +Y up.
        let ndc_x = 2.0 * u / width - 1.0;
        let ndc_y = 1.0 - 2.0 * v / height;

        let tan_half = (self.fov_y() * 0.5).tan();
        let x = ndc_x * (width / height) * tan_half;
        let y = ndc_y * tan_half;

        // Camera-space forward is +Z in this viewer convention.
        let dir_camera = Vec3::new(x, y, 1.0).to_normalized();
        let dir_world = self.camera_rotation().rotate_vec(dir_camera).to_normalized();

        (self.position(), dir_world)
    }
}


// =========================================================================
// Orbital Camera
// =========================================================================

pub struct OrbitalCamera {
    pub center: Vec3,
    pub rotation: Quaternion,
    pub radius: f32,

    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
    
    // For debug visualization: last orbit rotation axis
    pub last_orbit_axis: Vec3,
}

impl Default for OrbitalCamera {
    fn default() -> Self {
        Self {
            center: Vec3::new_zero(),
            rotation: Quaternion::new_identity(),
            radius: 10.0,
            fov_y: 45.0f32.to_radians(),
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
            last_orbit_axis: Vec3::new(0.0, 0.0, 1.0),
        }
    }
}

impl Camera for OrbitalCamera {
    fn view_matrix(&self) -> Mat4 {
        let rot_inv = self.rotation.inverse().to_matrix();
        let trans = Mat4::new_translation(-self.position());
        rot_inv * trans
    }
    fn camera_rotation(&self) -> Quaternion {
        self.rotation
    }

    fn projection_matrix(&self) -> Mat4 {
        Mat4::new_perspective_lh(self.fov_y, self.aspect, self.near, self.far)
    }

    fn position(&self) -> Vec3 {
        self.center - self.rotation.rotate_vec(Vec3::new(0.0, 0.0, self.radius))
    }

    fn target(&self) -> Vec3 {
        self.center
    }

    fn up(&self) -> Vec3 {
        self.rotation.rotate_vec(Vec3::new(0.0, 1.0, 0.0))
    }

    fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        // Calculate rotation axis in world space
        let local_axis = Vec3::new(delta_y, delta_x, 0.0);
        if local_axis.magnitude() < 1e-8 {
            return;
        }
        let rot_axis = self.rotation.rotate_vec(local_axis.to_normalized());
        let len = (delta_x * delta_x + delta_y * delta_y).sqrt();
        
        // Store the axis for debug visualization
        self.last_orbit_axis = rot_axis;
        
        self.rotation = Quaternion::from_axis_angle(rot_axis.to_normalized(), len) * self.rotation;
        self.rotation = self.rotation.to_normalized();
    }

    fn pan(&mut self, delta: Vec2) {
        // Pan moves the center.
        // Move along local Right and Up.
        let scale = self.radius * 0.1; // Adjust pan speed based on distance
        let right = self.rotation.rotate_vec(Vec3::new(1.0, 0.0, 0.0)) * -scale;
        let up = self.rotation.rotate_vec(Vec3::new(0.0, 1.0, 0.0)) * scale;

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

    fn look_at(&mut self, eye: Vec3, target: Vec3, _up: Vec3) {
        self.center = target;

        let dir = eye - target;
        self.radius = dir.magnitude();

        // Rotation maps +Z to forward(target-eye).
        let forward = (target - eye).to_normalized();
        self.rotation = Quaternion::from_unit_vecs(Vec3::new(0.0, 0.0, 1.0), forward).to_normalized();
    }

}
