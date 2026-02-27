use nalgebra::{
    Isometry3, Matrix4, Orthographic3, Perspective3, Point3, Unit, UnitQuaternion, Vector2, Vector3, Vector4
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

        // Screen center origin assumed
        let ndc_x = -1.0 + 2.0 * u / width;
        let ndc_y = 1.0 - 2.0 * v / height; 

        // D3D / Metal depth range
        let point_ndc_near = Point3::new(ndc_x, ndc_y, -1.0).to_homogeneous();
        let point_ndc_far  = Point3::new(ndc_x, ndc_y, 0.0).to_homogeneous();

        let world_near = inv_vp * point_ndc_near;
        let world_far  = inv_vp * point_ndc_far;

        let p_near = world_near.xyz() / world_near.w;
        let p_far  = world_far.xyz()  / world_far.w;

        let camera_pos = self.position();

        // if p_far and p_near and eye are not on the same line, it is error.
        if (p_far - camera_pos.coords).normalize().dot(&(p_near - camera_pos.coords).normalize()) < 0.999 {
            eprintln!("Warning: ray_from_screen may be inaccurate due to non-linear projection. Consider using a linear projection for accurate picking.");
        }
        println!("Ray from screen: near {:?}, far {:?}, camera_pos {:?}", p_near, p_far, camera_pos);

        let ray_origin = lin_alg::f32::Vec3::new(
            camera_pos.x,
            camera_pos.y,
            camera_pos.z,
        );

        let ray_direction = (p_far - camera_pos.coords).normalize();

        (
            ray_origin,
            lin_alg::f32::Vec3::new(
                -ray_direction.x,
                ray_direction.y,
                ray_direction.z,
            ),
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
        }
    }
}

impl OrbitalCamera {
    fn view_to_world(&self) -> Isometry3<f32> {
        let eye = self.position();
        let target = self.target();
        let up = self.up();
        Isometry3::look_at_rh(&eye, &target, &up)
    }
    fn world_to_view(&self) -> Isometry3<f32> {
        self.view_to_world().inverse()
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
        Perspective3::new(self.aspect, self.fov_y, self.near, self.far).to_homogeneous()
    }

    fn position(&self) -> Point3<f32> {
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
        // let local_rot_axis = self.view_to_world() * (delta_x * Vector3::y() + delta_y * Vector3::x());
        // let rot = UnitQuaternion::from_axis_angle(&Unit::new_normalize(local_rot_axis), local_rot_axis.magnitude());
        // self.rotation = rot * self.rotation;
        if delta_x < delta_y{
            // let rot_y = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), delta_x);
            let rot_x = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), delta_y);
            self.rotation =   rot_x *self.rotation;
        }else{
            let rot_y = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), delta_x);
            // let rot_x = UnitQuaternion::from_axis_angle(&Vector3::x_axis(), delta_y);
            self.rotation =   rot_y *self.rotation;
        }
       

    }

    fn pan(&mut self, delta: Vector2<f32>) {
        // Pan moves the center.
        // Move along local Right and Up.
        let scale = self.radius * 0.01; // Adjust pan speed based on distance
        let right = self.rotation * Vector3::x() * scale;
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
