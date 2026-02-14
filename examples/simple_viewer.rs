use graphics::{
    ControlScheme, EngineUpdates, EntityUpdate, GraphicsSettings, Scene, UiSettings, run,
    winit::event::{ElementState, MouseButton, WindowEvent},
};
use lin_alg::f32::Vec3;
use moleucle_3dview_rs::{Molecule, MoleculeViewer, viewer::ViewerEvent};
use std::path::Path;

fn main() {
    // 1. Initialize Viewer State
    let mut viewer = MoleculeViewer::new();

    // Load default molecule
    let path = Path::new("Benzene.mol2");
    if path.exists() {
        if let Ok(mol) = Molecule::from_mol2(path) {
            println!("Loaded molecule with {} atoms", mol.atoms.len());
            viewer.set_molecule(mol);
        } else {
            eprintln!("Failed to parse Benzene.mol2");
        }
    } else {
        eprintln!("Benzene.mol2 not found at {:?}", std::env::current_dir());
    }

    // 2. Initialize Scene
    let mut scene = Scene::default();

    // Configure Camera
    scene.camera.position = Vec3::new(0.0, 0.0, -10.0);
    // ControlScheme::Arc centers around 'center'
    scene.input_settings.control_scheme = ControlScheme::Arc {
        center: Vec3::new(0.0, 0.0, 0.0),
    };

    // 3. Initial Mesh Generation
    viewer.update_scene(&mut scene);

    // 4. Settings
    let ui_settings = UiSettings::default();
    let graphics_settings = GraphicsSettings::default();

    // 5. Run Application
    run(
        viewer,
        scene,
        ui_settings,
        graphics_settings,
        // Render Handler
        |viewer, scene, _dt| {
            if viewer.dirty {
                viewer.update_scene(scene);
                EngineUpdates {
                    meshes: true,
                    entities: EntityUpdate::All,
                    ..Default::default()
                }
            } else {
                EngineUpdates::default()
            }
        },
        // Device Event Handler
        |_viewer, _event, _scene, _is_synthetic, _dt| EngineUpdates::default(),
        // Window Event Handler
        |viewer, event, scene, _dt| {
            match event {
                WindowEvent::CursorMoved { position, .. } => {
                    viewer.on_mouse_move((position.x as f32, position.y as f32));
                }
                WindowEvent::MouseInput { state, button, .. } => {
                    if state == ElementState::Pressed && button == MouseButton::Left {
                        // Perform picking
                        let (near, far) = scene.screen_to_render(viewer.last_mouse_pos);
                        let dir = (far - near).to_normalized();

                        if let Some(event) = viewer.pick(near, dir) {
                            match event {
                                ViewerEvent::AtomClicked(i) => {
                                    println!("Callback: Atom {} Clicked", i)
                                }
                                ViewerEvent::BondClicked(i) => {
                                    println!("Callback: Bond {} Clicked", i)
                                }
                                ViewerEvent::NothingClicked => {
                                    println!("Callback: Nothing Clicked")
                                }
                            }
                        }
                    }
                }
                _ => {}
            }
            EngineUpdates::default()
        },
        // GUI Handler
        |viewer, ctx, scene| {
            egui::Window::new("Controls").show(ctx, |ui| {
                ui.label("Molecule Viewer");
                if let Some(mol) = &viewer.molecule {
                    ui.label(format!("Atoms: {}", mol.atoms.len()));
                    ui.label(format!("Bonds: {}", mol.bonds.len()));
                }

                if ui.button("Reset Camera").clicked() {
                    scene.camera.position = Vec3::new(0.0, 0.0, -10.0);
                    // Resetting look_at might require more logic depending on camera implementation
                    // But usually setting position and having Arc scheme is enough to reset view relative to center.
                    return EngineUpdates {
                        camera: true,
                        ..Default::default()
                    };
                }
                EngineUpdates::default()
            });
            EngineUpdates::default()
        },
    );
}
