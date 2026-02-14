use egui::Rect;
use nalgebra::{Isometry3, Matrix4, Orthographic3, Perspective3, Point2, Point3, Vector2, Vector3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ProjectionType {
    Perspective,
    Orthographic,
}

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Point3<f32>,
    pub target: Point3<f32>,
    pub up: Vector3<f32>,

    pub projection_type: ProjectionType,

    // Perspective
    pub fov: f32, // in radians

    // Orthographic
    pub ortho_scale: f32, // Height of view volume in world units

    pub aspect: f32,
    pub near: f32,
    pub far: f32,
}

impl Default for Camera {
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

impl Camera {
    pub fn view_matrix(&self) -> Matrix4<f32> {
        Isometry3::look_at_rh(&self.position, &self.target, &self.up).to_homogeneous()
    }

    pub fn projection_matrix(&self) -> Matrix4<f32> {
        match self.projection_type {
            ProjectionType::Perspective => {
                Perspective3::new(self.aspect, self.fov, self.near, self.far).to_homogeneous()
            }
            ProjectionType::Orthographic => {
                let width = self.ortho_scale * self.aspect;
                let height = self.ortho_scale;
                // nalgebra Orthographic3::new(left, right, bottom, top, near, far)
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

    pub fn view_projection(&self) -> Matrix4<f32> {
        self.projection_matrix() * self.view_matrix()
    }

    /// Projects a 3D point to 2D screen coordinates within `rect`.
    /// Returns `None` if the point is behind the camera.
    pub fn project_to_screen(&self, point: Point3<f32>, rect: Rect) -> Option<Point2<f32>> {
        let vp = self.view_projection();
        let p_clip = vp.transform_point(&point);
        let w = vp * point.to_homogeneous();
        let w = w.w; // Perspective division factor

        // In Orthographic, w is 1.0. In Perspective, it varies.
        // Clipping check:
        // OpenGL clip space: -w <= x,y,z <= w
        // But for "behind camera" in perspective, w > 0 usually logic.
        // For Ortho, w=1, so check z?

        if self.projection_type == ProjectionType::Perspective && w <= 0.0 {
            return None; // Behind camera
        }

        let ndc = p_clip.coords / w; // Normalized Device Coordinates (-1 to 1)

        // Map NDC to screen rect
        let x = rect.left() + (ndc.x + 1.0) * 0.5 * rect.width();
        let y = rect.top() + (1.0 - ndc.y) * 0.5 * rect.height(); // Y is down in egui

        Some(Point2::new(x, y))
    }

    // Zoom logic
    pub fn zoom(&mut self, delta: f32) {
        match self.projection_type {
            ProjectionType::Perspective => {
                let dir = self.target - self.position;
                let dist = dir.magnitude();
                let new_dist = (dist - delta).max(1.0); // Prevent zooming past target
                self.position = self.target - dir.normalize() * new_dist;
            }
            ProjectionType::Orthographic => {
                self.ortho_scale = (self.ortho_scale - delta).max(1.0);
            }
        }
    }

    // Rotate logic (orbit around target)
    pub fn rotate_orbit(&mut self, delta_x: f32, delta_y: f32) {
        let to_pos = self.position - self.target;

        // 1. Calculate local axes based on current orientation
        let fwd = -to_pos.normalize();
        let right = fwd.cross(&self.up).normalize();
        // Recalculate up to ensure it is perfectly orthogonal to fwd and right
        let local_up = right.cross(&fwd).normalize();

        // 2. Horizontal rotation (around current camera UP)
        let rot_y =
            nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(local_up), delta_x);
        let mut to_pos = rot_y * to_pos;
        let mut new_up = rot_y * local_up;

        // 3. Vertical rotation (around current camera RIGHT)
        let fwd = -to_pos.normalize();
        let right = fwd.cross(&new_up).normalize();
        let rot_x =
            nalgebra::Rotation3::from_axis_angle(&nalgebra::Unit::new_normalize(right), -delta_y);

        to_pos = rot_x * to_pos;
        new_up = rot_x * new_up;

        self.position = self.target + to_pos;
        self.up = new_up;
    }

    pub fn dolly(&mut self, delta: f32) {
        let dir = self.target - self.position;
        let dist = dir.magnitude();
        let new_dist = (dist - delta).max(1.0); // Prevent zooming past target
        self.position = self.target - dir.normalize() * new_dist;
    }

    pub fn pan(&mut self, delta: Vector2<f32>) {
        let fwd = (self.target - self.position).normalize();
        let right = fwd.cross(&self.up).normalize();
        let local_up = right.cross(&fwd).normalize();

        // Adjust pan speed based on zoom/distance
        let factor = match self.projection_type {
            ProjectionType::Perspective => (self.position - self.target).magnitude() * 0.001, // Adjusted heuristic
            ProjectionType::Orthographic => self.ortho_scale * 0.001,
        };

        let movement = right * delta.x * factor + local_up * delta.y * factor;
        self.position += movement;
        self.target += movement;
        self.up = local_up; // Ensure up stays orthogonal
    }

    /// Generates a ray for picking.
    /// u, v: screen coordinates in pixels.
    /// width, height: viewport dimensions in pixels.
    pub fn ray_from_screen(
        &self,
        u: f32,
        v: f32,
        width: f32,
        height: f32,
    ) -> (lin_alg::f32::Vec3, lin_alg::f32::Vec3) {
        let ndc_x = 2.0 * u / width - 1.0;
        let ndc_y = 1.0 - 2.0 * v / height;

        let fwd = (self.target - self.position).normalize();
        let right = fwd.cross(&self.up).normalize();
        let local_up = right.cross(&fwd).normalize();

        let ray_dir = match self.projection_type {
            ProjectionType::Perspective => {
                let tan_fov = (self.fov * 0.5).tan();
                (fwd + right * ndc_x * self.aspect * tan_fov + local_up * ndc_y * tan_fov)
                    .normalize()
            }
            ProjectionType::Orthographic => fwd,
        };

        let ray_origin = match self.projection_type {
            ProjectionType::Perspective => self.position,
            ProjectionType::Orthographic => {
                let w = self.ortho_scale * self.aspect;
                let h = self.ortho_scale;
                self.position + right * ndc_x * (w * 0.5) + local_up * ndc_y * (h * 0.5)
            }
        };

        (
            lin_alg::f32::Vec3::new(ray_origin.x, ray_origin.y, ray_origin.z),
            lin_alg::f32::Vec3::new(ray_dir.x, ray_dir.y, ray_dir.z),
        )
    }
}
