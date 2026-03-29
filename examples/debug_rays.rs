/// Example demonstrating debug ray visualization from camera to screen
/// This example shows how to use DebugRender to visualize picking rays for debugging

use graphics::{run, EngineUpdates, EntityUpdate, GraphicsSettings, Scene, UiSettings};
use lin_alg::f32::Vec3;
use moleucle_3dview_rs::{
    camera, viewer::ViewerEvent, CameraController, DebugRender, Molecule, MoleculeViewer,
};
use std::path::Path;

fn main() {
    // 1. Initialize State
    let mut viewer = MoleculeViewer::<DebugRender>::new();
    let controller = CameraController::<camera::OrbitalCamera>::new();

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

    // Initialize debug render with a default ray
    let initial_ray = (Vec3::new(0.0, 0.0, 10.0), Vec3::new(0.0, 0.0, -1.0));
    let mut debug_render = DebugRender::new(initial_ray);
    debug_render.set_ray_length(50.0);
    debug_render.set_ray_color((0.0, 1.0, 0.0)); // Green ray for visualization

    viewer.additional_render = Some(Box::new(debug_render));

    // 2. Initialize Scene
    let mut scene = Scene::default();

    // Initial Mesh Generation
    viewer.update_scene(&mut scene);

    // 3. Run Application
    run(
        // Wrap viewer and controller in a tuple
        (viewer, controller),
        scene,
        UiSettings::default(),
        GraphicsSettings::default(),
        // Render Handler
        |(viewer, controller), scene, _dt| {
            let mut updates = EngineUpdates::default();

            if viewer.dirty {
                viewer.update_scene(scene);
                updates.meshes = true;
                updates.entities = EntityUpdate::All;
            }

            // Controller handles camera info generation
            controller.update_scene_camera(scene);
            updates.camera = true;

            updates
        },
        // Device Event Handler
        |_state, _event, _scene, _is_synthetic, _dt| EngineUpdates::default(),
        // Window Event Handler
        |(viewer, controller), event, scene, _dt| {
            let (picked, updates) = controller.handle_event(&event, scene, viewer);

            if let Some(event) = picked {
                match &event {
                    ViewerEvent::AtomClicked(i) => {
                        println!("Atom {} clicked", i);
                        viewer.dirty = true;
                    }
                    ViewerEvent::BondClicked(i) => println!("Bond {} clicked", i),
                    ViewerEvent::NothingClicked => {} // Silently ignore clicks on nothing
                }
            }

            viewer.additional_render.as_mut().map(|debug| {
                let dir = controller.camera.last_orbit_axis;
                let dir:Vec3 = Vec3::new(dir.x, dir.y, dir.z);
                let origin = Vec3::new(0.0, 0.0, 0.0);
                debug.update_ray((origin, dir));

            });
            updates
        },
        // GUI Handler
        |(viewer, _controller), ctx, _scene| {
            egui::Window::new("Debug Ray Viewer").show(ctx, |ui| {
                ui.label("Molecule Viewer with Ray Visualization");
                if let Some(mol) = &viewer.molecule {
                    ui.label(format!("Atoms: {}", mol.atoms.len()));
                    ui.label(format!("Bonds: {}", mol.bonds.len()));
                }

                ui.separator();
                ui.label("Controls:");
                ui.label("Right Click: Orbit camera");
                ui.label("Middle Click + Shift: Pan camera");
                ui.label("Scroll: Zoom (dolly)");
                ui.label("Left Click: Pick atoms/bonds");
                ui.separator();
                ui.label("The green ray shows the picking");
                ui.label("ray direction from camera to screen");
            });
            EngineUpdates::default()
        },
    );
}
