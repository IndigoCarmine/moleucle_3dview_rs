use nalgebra::{
    Isometry3, Matrix4, Orthographic3, Perspective3, Point3, UnitQuaternion, Vector2, Vector3,
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

    fn position(&self) -> Point3<f32>;
    fn target(&self) -> Point3<f32>;
    fn up(&self) -> Vector3<f32>;

    fn set_aspect(&mut self, aspect: f32);

    fn orbit(&mut self, delta_x: f32, delta_y: f32);
    fn pan(&mut self, delta: Vector2<f32>);
    fn dolly(&mut self, delta: f32);

    fn fov(&self) -> f32;
    fn near(&self) -> f32;
    fn far(&self) -> f32;

    // Optional helper to set look_at if possible, otherwise it might be specific implementation dependent
    fn look_at(&mut self, eye: Point3<f32>, target: Point3<f32>, up: Vector3<f32>);

    fn ray_from_screen(
        &self,
        u: f32,
        v: f32,
        width: f32,
        height: f32,
    ) -> (lin_alg::f32::Vec3, lin_alg::f32::Vec3) {
        let ndc_x = 2.0 * u / width - 1.0;
        let ndc_y = 1.0 - 2.0 * v / height;

        let _fwd = (self.target() - self.position()).normalize();
        let _right = _fwd.cross(&self.up()).normalize();
        let _local_up = _right.cross(&_fwd).normalize();

        // Default to Perspective ray casting logic for now, or use projection matrix inverse
        // But projection matrix inverse is more generic.
        // Let's stick to the manual calculation assuming perspective as it's common.
        // Or better, use the inv_view_proj if we want to be generic.
        // For simplicity, let's copy the logic but adapt to generic Fov/Aspect.

        let ray_dir = {
            // Assume perspective for ray casting for now as it's the primary use case
            let _tan_fov = (self.fov() * 0.5).tan();
            // TODO: Ensure aspect is correct
            // self.aspect is not in trait, but projection matrix has it.
            // Let's assume aspect is handled by implementations or self.projection_matrix().

            // Re-deriving aspect from projection matrix (1,1) element?
            // Better to rely on the implementation specifics or keep it simple.

            // Actually, we can just use the provided view/proj matrices.
            let inv_vp = self
                .view_projection()
                .try_inverse()
                .unwrap_or_else(Matrix4::identity);

            // NDC near and far
            let point_ndc_near = Point3::new(ndc_x, ndc_y, -1.0).to_homogeneous();
            let point_ndc_far = Point3::new(ndc_x, ndc_y, 1.0).to_homogeneous();

            let point_world_near = inv_vp * point_ndc_near;
            let point_world_far = inv_vp * point_ndc_far;

            let p_near = point_world_near.xyz() / point_world_near.w;
            let p_far = point_world_far.xyz() / point_world_far.w;

            (p_far - p_near).normalize()
        };

        // Origin is position for perspective, or near plane point for ortho
        // The unproject method above handles both cases implicitly if inv_vp is correct.
        let ray_origin = self.position();

        (
            lin_alg::f32::Vec3::new(ray_origin.x, ray_origin.y, ray_origin.z),
            lin_alg::f32::Vec3::new(ray_dir.x, ray_dir.y, ray_dir.z),
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

    pub fov: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for OrbitalCamera {
    fn default() -> Self {
        Self {
            center: Point3::origin(),
            rotation: UnitQuaternion::identity(),
            radius: 10.0,
            fov: 45.0f32.to_radians(),
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
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

    fn projection_matrix(&self) -> Matrix4<f32> {
        Perspective3::new(self.aspect, self.fov, self.near, self.far).to_homogeneous()
    }

    fn position(&self) -> Point3<f32> {
        // Assume rotation identity -> looking at -Z?
        // Standard Orbital:
        // Eye = Center + Rotation * (0, 0, Radius)
        // This means at Identity, Eye is at (0,0,R) relative to center.
        // If we want to look at Center, we look along -Z.
        // So this matches standard RH coordinate system where +Z is out of screen.
        self.center + self.rotation * Vector3::new(0.0, 0.0, self.radius)
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
        // delta_x: yaw (around Y), delta_y: pitch (around local X/Right)

        let rot_y = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), delta_x);
        let rot_x = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), -delta_y);

        // Apply rotations: World Y then Local X?
        // self.rotation = rot_y * self.rotation * rot_x;

        // Let's try to be careful.
        // We want to Rotate around TARGET.
        // If we just rotate the 'rotation' quaternion:
        // New Rotation = RotY * CurrentRotation * RotX
        // This applies RotY in world space, and RotX in local space.
        self.rotation = rot_y * self.rotation * rot_x;
    }

    fn pan(&mut self, delta: Vector2<f32>) {
        // Pan moves the center.
        // Move along local Right and Up.
        let right = self.rotation * Vector3::x();
        let up = self.rotation * Vector3::y();

        self.center += right * delta.x + up * delta.y;
    }

    fn dolly(&mut self, delta: f32) {
        self.radius = (self.radius - delta).max(0.1);
    }

    fn fov(&self) -> f32 {
        self.fov
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

        // We need Rotation s.t. Rotation * (0,0,1) = dir_normalized
        // and Rotation * (0,1,0) ~= up
        // This is basically the inverse of look_at rotation.
        // LookAt mat has R rows as Right, Up, -Forward.
        // Here we want Rotation that transforms local (0,0,1) to Forward(away from target).
        // Standard LookAt from target to eye is: fwd = (eye-target).normalize().
        // So we want our local Z to map to fwd.

        self.rotation = UnitQuaternion::look_at_rh(&(target - eye), &up).inverse();

        // Wait, nalgebra look_at construction:
        // Eye at origin looking at 'direction' with 'up'.
        // We want orientation of the camera frame.
        // Camera frame has -Z as looking direction.
        // So local +Z is behind camera (towards eye from target).
        // eye - target is +Z direction in camera frame.
        // So we want Rotation that maps (0,0,1) to (eye-target).norm().

        let _fwd = (eye - target).normalize(); // This is global +Z direction for camera
                                               // We want a rotation that takes local +Z to global fwd, and local +Y to global Up (approx).

        // Quaternion::face_towards looks down -Z?
        // Let's use `UnitQuaternion::from_basis_unchecked` if we can build basis.
        // Or cleaner: Isometry look_at gives World-to-Camera. We want Camera-to-World rotation.
        let iso = Isometry3::look_at_rh(&eye, &target, &up);
        self.rotation = iso.rotation.inverse();
    }
}

// =========================================================================
// Legacy / LookAt Camera
// =========================================================================

#[derive(Clone, Copy, Debug)]
pub struct LookAtCamera {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,

    pub projection_type: ProjectionType,

    // Perspective
    pub fov: f32,

    // Orthographic
    pub ortho_scale: f32,

    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for LookAtCamera {
    fn default() -> Self {
        Self {
            position: Point3::new(0.0, 0.0, 10.0),
            target: Point3::origin(),
            up: Vector3::y(),
            projection_type: ProjectionType::Perspective,
            fov: 45.0f32.to_radians(),
            ortho_scale: 10.0,
            aspect: 1.0,
            near: 0.1,
            far: 100.0,
        }
    }
}

impl Camera for LookAtCamera {
    fn view_matrix(&self) -> Matrix4<f32> {
        Isometry3::look_at_rh(&self.position, &self.target, &self.up).to_homogeneous()
    }

    fn projection_matrix(&self) -> Matrix4<f32> {
        match self.projection_type {
            ProjectionType::Perspective => {
                Perspective3::new(self.aspect, self.fov, self.near, self.far).to_homogeneous()
            }
            ProjectionType::Orthographic => {
                let width = self.ortho_scale * self.aspect;
                let height = self.ortho_scale;
                Orthographic3::new(
                    -width / 2.0,
                    width / 2.0,
                    -height / 2.0,
                    height / 2.0,
                    self.near,
                    self.far,
                )
                .to_homogeneous()
            }
        }
    }

    fn position(&self) -> Point3<f32> {
        self.position
    }
    fn target(&self) -> Point3<f32> {
        self.target
    }
    fn up(&self) -> Vector3<f32> {
        self.up
    }

    fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        let to_pos = self.position - self.target;

        let fwd = -to_pos.normalize();
        let right = fwd.cross(&self.up).normalize();
        let local_up = right.cross(&fwd).normalize();

        // Horizontal rotation (around world/local UP?)
        // Replicating original behavior: around local_up
        let rot_y =
            nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(local_up), delta_x);
        let mut to_pos = rot_y * to_pos;
        let mut new_up = rot_y * local_up;

        // Vertical
        let fwd = -to_pos.normalize();
        let right = fwd.cross(&new_up).normalize();
        let rot_x =
            nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(right), -delta_y);

        to_pos = rot_x * to_pos;
        new_up = rot_x * new_up;

        self.position = self.target + to_pos;
        self.up = new_up;
    }

    fn pan(&mut self, delta: Vector2<f32>) {
        let fwd = (self.target - self.position).normalize();
        let right = fwd.cross(&self.up).normalize();
        let local_up = right.cross(&fwd).normalize();

        // Adjust pan speed based on zoom/distance
        let factor = match self.projection_type {
            ProjectionType::Perspective => (self.position - self.target).magnitude() * 0.001,
            ProjectionType::Orthographic => self.ortho_scale * 0.001,
        };

        let movement = right * delta.x * factor + local_up * delta.y * factor;
        self.position += movement;
        self.target += movement;
        self.up = local_up;
    }

    fn dolly(&mut self, delta: f32) {
        match self.projection_type {
            ProjectionType::Perspective => {
                let dir = self.target - self.position;
                let dist = dir.magnitude();
                let new_dist = (dist - delta).max(1.0);
                self.position = self.target - dir.normalize() * new_dist;
            }
            ProjectionType::Orthographic => {
                self.ortho_scale = (self.ortho_scale - delta).max(1.0);
            }
        }
    }

    fn fov(&self) -> f32 {
        self.fov
    }
    fn near(&self) -> f32 {
        self.near
    }
    fn far(&self) -> f32 {
        self.far
    }

    fn look_at(&mut self, eye: Point3<f32>, target: Point3<f32>, up: Vector3<f32>) {
        self.position = eye;
        self.target = target;
        self.up = up;
    }

    // Override ray_from_screen if existing logic was special?
    // The default impl in trait uses Isometry inverse, which should match.
    // We can rely on default.
}
