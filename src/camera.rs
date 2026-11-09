use lin_alg::f32::{Mat4, Quaternion, Vec2, Vec3};

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

    /// Set the near and far clip planes directly.
    fn set_clip_planes(&mut self, near: f32, far: f32);

    /// Fit the near and far planes around a bounding sphere of the scene, so
    /// that nothing inside it is clipped however far the camera is from it.
    ///
    /// From outside the sphere both planes hug it, which spends the depth
    /// buffer's precision on the scene alone. Once the camera is inside it,
    /// the near plane is held at `far / MAX_CLIP_RATIO`, so the depth range
    /// stays usable while geometry right in front of the camera still shows.
    fn fit_clip_planes(&mut self, center: Vec3, radius: f32) {
        let radius = if radius.is_finite() {
            radius.max(0.0)
        } else {
            0.0
        };
        let distance = (self.position() - center).magnitude();
        if !distance.is_finite() {
            return;
        }

        // Small slack so geometry exactly on the sphere is not clipped by
        // rounding.
        let far = (distance + radius) * 1.02 + 0.01;
        let near = ((distance - radius) * 0.98)
            .max(far / MAX_CLIP_RATIO)
            .max(MIN_NEAR);

        self.set_clip_planes(near, far.max(near * 2.0));
    }

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
            // Only a fallback: viewports refit both planes to the scene every
            // frame with `fit_clip_planes`. Kept generous so a camera used
            // without that does not silently drop distant geometry.
            far: 10_000.0,
            last_orbit_axis: Vec3::new(0.0, 0.0, 1.0),
        }
    }
}

/// Largest far/near ratio `fit_clip_planes` allows. A 24-bit depth buffer
/// with a perspective projection loses most of its precision to the region
/// near the camera; past roughly this ratio the far side starts z-fighting.
const MAX_CLIP_RATIO: f32 = 1.0e4;

/// Smallest near plane `fit_clip_planes` will set, in scene units (nm).
const MIN_NEAR: f32 = 1.0e-3;

impl OrbitalCamera {
    /// An orbital camera that shows exactly what `fly` shows, orbiting a
    /// point `radius` in front of it.
    pub fn from_fly(fly: &FlyCamera, radius: f32) -> Self {
        let radius = if radius.is_finite() { radius.max(0.1) } else { 10.0 };
        Self {
            center: fly.position + fly.forward() * radius,
            rotation: fly.rotation,
            radius,
            fov_y: fly.fov_y,
            aspect: fly.aspect,
            near: fly.near,
            far: fly.far,
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
    fn set_clip_planes(&mut self, near: f32, far: f32) {
        self.near = near;
        self.far = far;
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

// =========================================================================
// Fly Camera
// =========================================================================

/// A first-person camera: it rotates in place around its own eye and moves
/// through the scene, rather than orbiting a target.
///
/// Uses the same conventions as [`OrbitalCamera`] (left-handed, camera-space
/// +Z forward), so the two can be swapped with [`FlyCamera::from_orbital`] and
/// [`OrbitalCamera::from_fly`] without the view jumping.
pub struct FlyCamera {
    pub position: Vec3,
    pub rotation: Quaternion,

    pub fov_y: f32,
    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for FlyCamera {
    fn default() -> Self {
        Self::from_orbital(&OrbitalCamera::default())
    }
}

impl FlyCamera {
    /// A fly camera at the orbital camera's eye, looking the same way.
    pub fn from_orbital(orbital: &OrbitalCamera) -> Self {
        Self {
            position: orbital.position(),
            rotation: orbital.rotation,
            fov_y: orbital.fov_y,
            aspect: orbital.aspect,
            near: orbital.near,
            far: orbital.far,
        }
    }

    /// World-space viewing direction.
    pub fn forward(&self) -> Vec3 {
        self.rotation.rotate_vec(Vec3::new(0.0, 0.0, 1.0))
    }

    /// World-space right direction.
    pub fn right(&self) -> Vec3 {
        self.rotation.rotate_vec(Vec3::new(1.0, 0.0, 0.0))
    }

    /// Turn in place: `yaw` radians about the camera's own up axis (positive
    /// turns right), then `pitch` radians about its right axis (positive looks
    /// down, matching screen-space mouse motion). No world up is enforced,
    /// matching [`OrbitalCamera::orbit`]'s free rotation.
    pub fn look(&mut self, yaw: f32, pitch: f32) {
        if yaw != 0.0 {
            let axis = self.up().to_normalized();
            self.rotation = Quaternion::from_axis_angle(axis, yaw) * self.rotation;
        }
        if pitch != 0.0 {
            let axis = self.right().to_normalized();
            self.rotation = Quaternion::from_axis_angle(axis, pitch) * self.rotation;
        }
        self.rotation = self.rotation.to_normalized();
    }

    /// Move by `local` in camera space: x = right, y = up, z = forward.
    pub fn translate_local(&mut self, local: Vec3) {
        self.position += self.rotation.rotate_vec(local);
    }
}

impl Camera for FlyCamera {
    fn view_matrix(&self) -> Mat4 {
        let rot_inv = self.rotation.inverse().to_matrix();
        let trans = Mat4::new_translation(-self.position);
        rot_inv * trans
    }

    fn projection_matrix(&self) -> Mat4 {
        Mat4::new_perspective_lh(self.fov_y, self.aspect, self.near, self.far)
    }

    fn camera_rotation(&self) -> Quaternion {
        self.rotation
    }

    fn position(&self) -> Vec3 {
        self.position
    }

    fn target(&self) -> Vec3 {
        self.position + self.forward()
    }

    fn up(&self) -> Vec3 {
        self.rotation.rotate_vec(Vec3::new(0.0, 1.0, 0.0))
    }

    fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    fn orbit(&mut self, delta_x: f32, delta_y: f32) {
        self.look(delta_x, delta_y);
    }

    fn pan(&mut self, delta: Vec2) {
        self.translate_local(Vec3::new(-delta.x, delta.y, 0.0));
    }

    fn dolly(&mut self, delta: f32) {
        self.translate_local(Vec3::new(0.0, 0.0, delta));
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
    fn set_clip_planes(&mut self, near: f32, far: f32) {
        self.near = near;
        self.far = far;
    }

    fn look_at(&mut self, eye: Vec3, target: Vec3, _up: Vec3) {
        self.position = eye;
        let forward = (target - eye).to_normalized();
        self.rotation =
            Quaternion::from_unit_vecs(Vec3::new(0.0, 0.0, 1.0), forward).to_normalized();
    }
}
