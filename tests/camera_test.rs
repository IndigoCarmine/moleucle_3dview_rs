use moleucle_3dview_rs::camera::{Camera, FlyCamera, OrbitalCamera};
use lin_alg::f32::{Vec2, Vec3};

#[test]
fn test_orbital_camera_look_at() {
    let mut cam = OrbitalCamera::default();

    let eye = Vec3::new(0.0, 0.0, -10.0);
    let target = Vec3::new_zero();
    let up = Vec3::new(0.0, 1.0, 0.0);

    cam.look_at(eye, target, up);

    // Check internal state
    assert_eq!(cam.center, target);
    assert!((cam.radius - 10.0).abs() < 1e-5);

    // Check position reconstruction
    let pos = cam.position();
    println!("Pos: {:?}", pos);
    assert!((pos - eye).magnitude() < 1e-5, "Position should match eye");

    // Check up vector reconstruction (approx)
    let cam_up = cam.up();
    assert!((cam_up - up).magnitude() < 1e-5, "Up vector should match");
}

#[test]
fn test_orbital_camera_pan() {
    let mut cam = OrbitalCamera::default(); // Center 0,0,0, R=10, Looking -Z? No default R=10 at (0,0,10) looking -Z.
                                            // Default: Center=0,0,0. Rot=Identity. Radius=10.
                                            // Pos = 0 + I * (0,0,10) = (0,0,10).
                                            // Target = 0,0,0.
                                            // Fwd (Screen In) = -Z.
                                            // Right = +X. Up = +Y.

    // Pan right (x=1.0).
    // center += Right * 1.0 = (1.0, 0, 0).
    cam.pan(Vec2::new(1.0, 0.0));

    assert!((cam.center - Vec3::new(-1.0, 0.0, 0.0)).magnitude() < 1e-5);

    // Position should also shift by (1,0,0)
    let pos = cam.position();
    // Original pos (0,0,-10). New should be (-1,0,-10).
    assert!((pos - Vec3::new(-1.0, 0.0, -10.0)).magnitude() < 1e-5);
}

#[test]
fn test_ray_cast_default() {
    let mut cam = OrbitalCamera::default();
    // Default cam at (0,0,-10) looking at (0,0,0). Fov 45 deg. Aspect 1.0.
    // Screen center (u=width/2, v=height/2) -> NDC (0,0).
    // Ray should be straight down -Z.

    let w = 800.0;
    let h = 600.0;
    cam.set_aspect(w / h);

    let (origin, dir) = cam.ray_from_screen(w / 2.0, h / 2.0, w, h);

    // Origin should be camera pos (0,0,-10) for perspective
    assert!((origin.x - 0.0).abs() < 1e-5);
    assert!((origin.y - 0.0).abs() < 1e-5);
    assert!((origin.z + 10.0).abs() < 1e-5);

    // Dir should be (0,0,1)
    assert!((dir.x - 0.0).abs() < 1e-5);
    assert!((dir.y - 0.0).abs() < 1e-5);
    assert!((dir.z - 1.0).abs() < 1e-5);
}

/// Project a world point and report whether it lies inside the depth range.
fn inside_depth_range(cam: &OrbitalCamera, p: Vec3) -> bool {
    let m = cam.view_projection().data;
    // Column-major: clip = M * (p, 1).
    let z = m[2] * p.x + m[6] * p.y + m[10] * p.z + m[14];
    let w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
    w > 0.0 && z >= 0.0 && z <= w
}

#[test]
fn fit_clip_planes_keeps_a_distant_scene_inside_the_depth_range() {
    let mut cam = OrbitalCamera::default();
    // A 500 nm system framed from 1000 nm away: far past the old fixed
    // 100 nm far plane.
    cam.center = Vec3::new_zero();
    cam.radius = 1000.0;
    cam.fit_clip_planes(Vec3::new_zero(), 500.0);

    let forward = cam.target() - cam.position();
    let forward = forward / forward.magnitude();
    let nearest = cam.target() - forward * 499.0;
    let farthest = cam.target() + forward * 499.0;
    assert!(inside_depth_range(&cam, nearest));
    assert!(inside_depth_range(&cam, farthest));
    assert!(cam.near < cam.far);
}

#[test]
fn fit_clip_planes_inside_the_scene_keeps_a_usable_depth_ratio() {
    let mut cam = OrbitalCamera::default();
    cam.center = Vec3::new_zero();
    cam.radius = 1.0;
    cam.fit_clip_planes(Vec3::new_zero(), 200.0);

    assert!(cam.near > 0.0);
    assert!(cam.far / cam.near <= 1.0e4 * 1.0001);
    let forward = cam.target() - cam.position();
    let forward = forward / forward.magnitude();
    assert!(inside_depth_range(&cam, cam.position() + forward * 199.0));
}

fn approx_vec(a: Vec3, b: Vec3, eps: f32) -> bool {
    (a - b).magnitude() < eps
}

fn tilted_orbital() -> OrbitalCamera {
    let mut cam = OrbitalCamera::default();
    cam.center = Vec3::new(1.0, -2.0, 3.0);
    cam.radius = 7.5;
    cam.orbit(0.4, -0.3);
    cam
}

#[test]
fn fly_from_orbital_shows_the_same_view() {
    let orbital = tilted_orbital();
    let fly = FlyCamera::from_orbital(&orbital);
    assert!(approx_vec(fly.position(), orbital.position(), 1e-4));
    let (a, b) = (fly.view_matrix().data, orbital.view_matrix().data);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-4, "{a:?} vs {b:?}");
    }
}

#[test]
fn orbital_fly_orbital_round_trip_keeps_the_view() {
    let orbital = tilted_orbital();
    let back = OrbitalCamera::from_fly(&FlyCamera::from_orbital(&orbital), orbital.radius);
    assert!(approx_vec(back.center, orbital.center, 1e-4));
    let (a, b) = (back.view_matrix().data, orbital.view_matrix().data);
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-4, "{a:?} vs {b:?}");
    }
}

#[test]
fn fly_look_turns_in_place() {
    let mut fly = FlyCamera::from_orbital(&tilted_orbital());
    let eye = fly.position();
    let forward = fly.forward();
    fly.look(0.3, -0.2);
    assert!(approx_vec(fly.position(), eye, 1e-6), "eye must not move");
    assert!(!approx_vec(fly.forward(), forward, 1e-3), "view must turn");
}

#[test]
fn fly_look_directions_follow_the_mouse() {
    let mut fly = FlyCamera::default();
    let right = fly.right();
    let up = fly.up();
    // Mouse right (positive yaw) turns toward the camera's right.
    fly.look(0.1, 0.0);
    assert!(fly.forward().dot(right) > 0.0);
    // Mouse down (positive pitch) looks down.
    let mut fly = FlyCamera::default();
    fly.look(0.0, 0.1);
    assert!(fly.forward().dot(up) < 0.0);
}

#[test]
fn fly_translate_local_moves_along_camera_axes() {
    let mut fly = FlyCamera::from_orbital(&tilted_orbital());
    let start = fly.position();
    let (forward, right, up) = (fly.forward(), fly.right(), fly.up());
    fly.translate_local(Vec3::new(1.0, 2.0, 3.0));
    let expected = start + right * 1.0 + up * 2.0 + forward * 3.0;
    assert!(approx_vec(fly.position(), expected, 1e-4));
}

#[test]
fn fit_clip_planes_works_for_the_fly_camera() {
    let mut fly = FlyCamera::default();
    fly.position = Vec3::new(0.0, 0.0, -1000.0);
    fly.fit_clip_planes(Vec3::new_zero(), 50.0);
    assert!(fly.near > 900.0 && fly.far > 1050.0, "{} {}", fly.near, fly.far);
}
