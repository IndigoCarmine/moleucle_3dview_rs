use crate::{
    additional_render::AdditionalRender,
    camera::Camera,
    viewer::{MoleculeViewer, ViewerEvent},
};
use graphics::winit::keyboard::{KeyCode, PhysicalKey};
use graphics::{
    winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent},
    EngineUpdates, Scene,
};
use lin_alg::f32::Vec2;

pub struct CameraController<T: Camera + Default> {
    pub camera: Box<T>,
    last_mouse_pos: Vec2,
    mouse_lb_pressed: bool,
    mouse_mb_pressed: bool,
    mouse_rb_pressed: bool,
    shift_pressed: bool,
    ctrl_pressed: bool,
    width: f32,
    height: f32,
}

impl<T: Camera + Default> CameraController<T> {
    pub fn new() -> Self {
        let width = 800.0;
        let height = 600.0;
        let mut camera = T::default();
        camera.set_aspect(width / height);

        Self {
            camera: Box::new(camera),
            last_mouse_pos: Vec2::new(0.0, 0.0),
            mouse_lb_pressed: false,
            mouse_mb_pressed: false,
            mouse_rb_pressed: false,
            shift_pressed: false,
            ctrl_pressed: false,
            width,
            height,
        }
    }

    /// Blender-style navigation:
    /// - MMB drag: orbit
    /// - Shift + MMB: pan
    /// - Ctrl + MMB: dolly
    /// - LMB: pick
    pub fn handle_event<U: AdditionalRender>(
        &mut self,
        event: &WindowEvent,
        _scene: &Scene,
        viewer: &MoleculeViewer<U>,
    ) -> (Option<ViewerEvent>, EngineUpdates) {
        let mut updates = EngineUpdates::default();
        let mut picked_event = None;

        match event {
            WindowEvent::Resized(size) => {
                self.width = size.width as f32;
                self.height = size.height as f32;
                self.camera.set_aspect(self.width / self.height);
                updates.camera = true;
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let PhysicalKey::Code(keycode) = event.physical_key {
                    match keycode {
                        KeyCode::ShiftLeft | KeyCode::ShiftRight => {
                            self.shift_pressed = pressed;
                        }
                        KeyCode::ControlLeft | KeyCode::ControlRight => {
                            self.ctrl_pressed = pressed;
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = *state == ElementState::Pressed;
                match button {
                    MouseButton::Left => {
                        self.mouse_lb_pressed = pressed;
                        if pressed {
                            // Picking
                            let (ray_origin, ray_dir) = self.camera.ray_from_screen(
                                self.last_mouse_pos.x,
                                self.last_mouse_pos.y,
                                self.width,
                                self.height,
                            );
                            picked_event = viewer.pick(ray_origin, ray_dir);
                        }
                    }
                    MouseButton::Middle => self.mouse_mb_pressed = pressed,
                    MouseButton::Right => self.mouse_rb_pressed = pressed,
                    _ => {}
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let new_pos = Vec2::new(position.x as f32, position.y as f32);
                let delta_x = new_pos.x - self.last_mouse_pos.x;
                let delta_y = new_pos.y - self.last_mouse_pos.y;

                // Orbit with MMB (or RMB for convenience)
                if self.mouse_mb_pressed || self.mouse_rb_pressed {
                    if self.shift_pressed {
                        // Pan
                        let sensitivity = 0.01;
                        self.camera
                            .pan(Vec2::new(delta_x * sensitivity, delta_y * sensitivity));
                    } else if self.ctrl_pressed {
                        // Dolly
                        self.camera.dolly(delta_y * 0.1);
                    } else {
                        // Orbit
                        // Sensitivity: 0.005 radians per pixel
                        self.camera.orbit(delta_x * 0.005, delta_y * 0.005);
                    }
                    updates.camera = true;
                }
                self.last_mouse_pos = new_pos;
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => *y * 1.0,
                    MouseScrollDelta::PixelDelta(pos) => pos.y as f32 * 0.1,
                };
                self.camera.dolly(scroll);
                updates.camera = true;
            }
            _ => {}
        }

        (picked_event, updates)
    }

    /// Synchronize camera state into rendering scene.
    pub fn update_scene_camera(&self, scene: &mut Scene) {
                let pos = self.camera.position();

        scene.camera.position = lin_alg::f32::Vec3::new(pos.x, pos.y, pos.z);
        let rot = self.camera.camera_rotation();
        scene.camera.orientation = lin_alg::f32::Quaternion::new(rot.w, rot.x, rot.y, rot.z);

        scene.camera.fov_y = self.camera.fov_y();
        scene.camera.near = self.camera.near();
        scene.camera.far = self.camera.far();
        // Aspect
        scene.camera.aspect = self.width / self.height;

        // Update the project matrix in the graphics engine
        scene.camera.update_proj_mat();
    }

    pub fn ray_from_last_mouse(&self) -> (lin_alg::f32::Vec3, lin_alg::f32::Vec3) {
        self.camera.ray_from_screen(
            self.last_mouse_pos.x,
            self.last_mouse_pos.y,
            self.width,
            self.height,
        )
    }
}
