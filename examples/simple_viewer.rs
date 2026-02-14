use graphics::{run, EngineUpdates, EntityUpdate, GraphicsSettings, Scene, UiSettings};
use moleucle_3dview_rs::{
    viewer::ViewerEvent, CameraController, Molecule, MoleculeViewer, SelectedAtomRender,
};
use std::path::Path;

fn main() {
    // 1. Initialize State
    let mut viewer = MoleculeViewer::new();
    let mut controller = CameraController::new();

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

    let selected_atom_render = SelectedAtomRender::new();
    viewer.additional_render = Some(Box::new(selected_atom_render));

    // 2. Initialize Scene
    let mut scene = Scene::default();

    // Sync initial camera state
    controller.camera.position = nalgebra::Point3::new(0.0, 0.0, -10.0);
    controller.update_scene_camera(&mut scene);

    // Initial Mesh Generation
    viewer.update_scene(&mut scene);

    // 3. Run Application
    run(
        // We need to pass both viewer and controller.
        // We can wrap them in a tuple or a struct.
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
                    ViewerEvent::AtomClicked(i) => println!("Main Trace: Atom {} Clicked", i),
                    ViewerEvent::BondClicked(i) => println!("Main Trace: Bond {} Clicked", i),
                    ViewerEvent::NothingClicked => println!("Main Trace: Nothing Clicked"),
                }

                if let Some(render) = &mut viewer.additional_render {
                    render.handle_event(&event);
                    viewer.dirty = true;
                }
            }

            updates
        },
        // GUI Handler
        |(viewer, _controller), ctx, _scene| {
            egui::Window::new("Controls").show(ctx, |ui| {
                ui.label("Molecule Viewer");
                if let Some(mol) = &viewer.molecule {
                    ui.label(format!("Atoms: {}", mol.atoms.len()));
                    ui.label(format!("Bonds: {}", mol.bonds.len()));
                }

                ui.separator();
                ui.label("Controls:");
                ui.label("Right Click: Orbit");
                ui.label("Middle Click: Pan");
                ui.label("Scroll: Zoom");
                ui.label("Left Click: Select");
            });
            EngineUpdates::default()
        },
    );
}
