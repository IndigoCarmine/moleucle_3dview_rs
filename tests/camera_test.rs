use moleucle_3dview_rs::camera::{Camera, OrbitalCamera};
use nalgebra::{Point3, Vector2, Vector3};

#[test]
fn test_orbital_camera_look_at() {
    let mut cam = OrbitalCamera::default();

    let eye = Point3::new(0.0, 0.0, -10.0);
    let target = Point3::origin();
    let up = Vector3::y();

    cam.look_at(eye, target, up);

    // Check internal state
    assert_eq!(cam.center, target);
    assert!((cam.radius - 10.0).abs() < 1e-5);

    // Check position reconstruction
    let pos = cam.position();
    println!("Pos: {:?}", pos);
    assert!((pos - eye).norm() < 1e-5, "Position should match eye");

    // Check up vector reconstruction (approx)
    let cam_up = cam.up();
    assert!((cam_up - up).norm() < 1e-5, "Up vector should match");
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
    cam.pan(Vector2::new(1.0, 0.0));

    assert!((cam.center - Point3::new(1.0, 0.0, 0.0)).norm() < 1e-5);

    // Position should also shift by (1,0,0)
    let pos = cam.position();
    // Original pos (0,0,10). New should be (1,0,10).
    assert!((pos - Point3::new(1.0, 0.0, 10.0)).norm() < 1e-5);
}

#[test]
fn test_ray_cast_default() {
    let mut cam = OrbitalCamera::default();
    // Default cam at (0,0,10) looking at (0,0,0). Fov 45 deg. Aspect 1.0.
    // Screen center (u=width/2, v=height/2) -> NDC (0,0).
    // Ray should be straight down -Z.

    let w = 800.0;
    let h = 600.0;
    cam.set_aspect(w / h);

    let (origin, dir) = cam.ray_from_screen(w / 2.0, h / 2.0, w, h);

    // Origin should be camera pos (0,0,10) for perspective
    assert!((origin.x - 0.0).abs() < 1e-5);
    assert!((origin.y - 0.0).abs() < 1e-5);
    assert!((origin.z - 10.0).abs() < 1e-5);

    // Dir should be (0,0,-1)
    assert!((dir.x - 0.0).abs() < 1e-5);
    assert!((dir.y - 0.0).abs() < 1e-5);
    assert!((dir.z + 1.0).abs() < 1e-5); // -(-1) = 1
}
