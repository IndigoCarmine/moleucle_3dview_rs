//! View-frustum test, used to skip periodic images that cannot be on screen.
//!
//! Without this, an off-screen image still costs a full run of the vertex
//! shader over every vertex it contains — the GPU only discards it afterwards,
//! at the clip stage. Replicating a large molecule 27 times would therefore
//! shade 27x the geometry no matter how far in the camera is zoomed.

use lin_alg::f32::Vec3;

/// Half-space `dot(normal, p) + d >= 0`, pointing inwards.
#[derive(Clone, Copy, Debug)]
struct Plane {
    normal: Vec3,
    d: f32,
}

impl Plane {
    /// Signed distance from the plane to `point`, positive on the inside.
    #[inline]
    fn distance(&self, point: Vec3) -> f32 {
        self.normal.dot(point) + self.d
    }
}

/// The six clipping planes of a view-projection matrix.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Frustum {
    planes: [Plane; 6],
}

impl Frustum {
    /// Extract the planes from a column-major view-projection matrix
    /// (Gribb & Hartmann): each plane is a sum or difference of two matrix rows,
    /// because a point is inside when its clip coordinates satisfy
    /// `-w <= x, y, z <= w`.
    ///
    /// This crate's projection uses the wgpu/D3D depth convention (`0 <= z <= w`
    /// rather than `-w <= z`), so the near plane is row 3 alone.
    pub(crate) fn from_view_proj(m: &[f32; 16]) -> Self {
        // Column-major: m[column * 4 + row]. Row `r` of the matrix is
        // (m[r], m[4 + r], m[8 + r], m[12 + r]).
        let row = |r: usize| [m[r], m[4 + r], m[8 + r], m[12 + r]];
        let (x, y, z, w) = (row(0), row(1), row(2), row(3));

        let plane = |a: [f32; 4]| {
            let normal = Vec3::new(a[0], a[1], a[2]);
            // Normalising keeps `distance` in world units, which is what makes
            // comparing it against a sphere radius meaningful.
            let length = normal.magnitude();
            if length > 1e-12 {
                Plane {
                    normal: normal / length,
                    d: a[3] / length,
                }
            } else {
                // A degenerate row cannot cull anything; make it accept.
                Plane {
                    normal: Vec3::new(0.0, 0.0, 0.0),
                    d: f32::MAX,
                }
            }
        };
        let combine = |a: [f32; 4], b: [f32; 4], add: bool| {
            let sign = if add { 1.0 } else { -1.0 };
            plane([
                a[0] + sign * b[0],
                a[1] + sign * b[1],
                a[2] + sign * b[2],
                a[3] + sign * b[3],
            ])
        };

        Self {
            planes: [
                combine(w, x, true),  // left:   w + x >= 0
                combine(w, x, false), // right:  w - x >= 0
                combine(w, y, true),  // bottom: w + y >= 0
                combine(w, y, false), // top:    w - y >= 0
                plane(z),             // near:   z >= 0
                combine(w, z, false), // far:    w - z >= 0
            ],
        }
    }

    /// Whether a sphere is at least partly inside.
    ///
    /// Conservative: a sphere straddling a plane counts as visible, and the
    /// corner regions outside two planes at once are not rejected. Both err
    /// towards drawing, which is the safe direction — a false positive costs a
    /// wasted draw, a false negative loses geometry the user should see.
    pub(crate) fn intersects_sphere(&self, center: Vec3, radius: f32) -> bool {
        self.planes
            .iter()
            .all(|plane| plane.distance(center) >= -radius)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lin_alg::f32::{Mat4, Vec3};

    /// A camera at the origin looking down +z, matching this crate's convention
    /// (see `OrbitalCamera`), with a 90-degree vertical field of view.
    fn view_proj() -> [f32; 16] {
        let near = 0.1_f32;
        let far = 100.0_f32;
        let f = 1.0_f32; // 1 / tan(45 degrees)

        // Column-major perspective for a +z-forward, [0, 1] depth range.
        Mat4::new([
            f, 0.0, 0.0, 0.0, //
            0.0, f, 0.0, 0.0, //
            0.0, 0.0, far / (far - near), 1.0, //
            0.0, 0.0, -(far * near) / (far - near), 0.0,
        ])
        .data
    }

    fn frustum() -> Frustum {
        Frustum::from_view_proj(&view_proj())
    }

    #[test]
    fn a_point_straight_ahead_is_inside() {
        assert!(frustum().intersects_sphere(Vec3::new(0.0, 0.0, 10.0), 0.0));
    }

    #[test]
    fn geometry_behind_the_camera_is_rejected() {
        assert!(!frustum().intersects_sphere(Vec3::new(0.0, 0.0, -10.0), 1.0));
    }

    #[test]
    fn geometry_past_the_far_plane_is_rejected() {
        assert!(!frustum().intersects_sphere(Vec3::new(0.0, 0.0, 500.0), 1.0));
    }

    #[test]
    fn geometry_off_to_the_side_is_rejected() {
        // At z = 10 with a 90-degree fov the frustum spans |x| <= 10.
        let f = frustum();
        assert!(f.intersects_sphere(Vec3::new(9.0, 0.0, 10.0), 0.5));
        assert!(!f.intersects_sphere(Vec3::new(60.0, 0.0, 10.0), 1.0));
    }

    /// The whole point of the radius: a sphere whose centre is outside but which
    /// still pokes into view has to be drawn.
    #[test]
    fn a_sphere_straddling_a_plane_counts_as_visible() {
        let f = frustum();
        let just_outside = Vec3::new(12.0, 0.0, 10.0);
        assert!(!f.intersects_sphere(just_outside, 0.1));
        assert!(f.intersects_sphere(just_outside, 5.0));
    }

    /// An enormous sphere contains the camera, so it is visible from anywhere.
    #[test]
    fn a_sphere_enclosing_the_camera_is_visible() {
        assert!(frustum().intersects_sphere(Vec3::new(0.0, 0.0, 0.0), 1000.0));
    }

    /// The identity matrix has to degrade to "accept", not to a frustum that
    /// culls everything -- `RenderFrameState::new` hands out identity view
    /// matrices in tests and headless paths.
    #[test]
    fn an_identity_matrix_does_not_cull_the_origin() {
        let mut identity = [0.0f32; 16];
        for i in 0..4 {
            identity[i * 4 + i] = 1.0;
        }
        assert!(Frustum::from_view_proj(&identity).intersects_sphere(Vec3::new(0.0, 0.0, 0.5), 1.0));
    }
}
