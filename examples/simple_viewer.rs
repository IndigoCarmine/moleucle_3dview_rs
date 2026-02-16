use graphics::{run, EngineUpdates, EntityUpdate, GraphicsSettings, Scene, UiSettings};
use moleucle_3dview_rs::{
    camera, viewer::ViewerEvent, Camera, CameraController, Molecule, MoleculeViewer,
    SelectedAtomRender,
};
use std::path::Path;

fn main() {
    // 1. Initialize State
    let mut viewer = MoleculeViewer::new();
    let mut controller = CameraController::<camera::OrbitalCamera>::new();

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

    viewer.additional_render = Some(Box::new(SelectedAtomRender::new()));

    // 2. Initialize Scene
    let mut scene = Scene::default();

    // Sync initial camera state
    controller.camera.look_at(
        nalgebra::Point3::new(0.0, 0.0, -10.0),
        nalgebra::Point3::origin(),
        nalgebra::Vector3::y(),
    );
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
                    ViewerEvent::AtomClicked(i) => {
                        println!("Main Trace: Atom {} Clicked", i);
                        if let Some(selected_atom) = &mut viewer.additional_render {
                            selected_atom.add_atom(*i);
                            viewer.dirty = true;
                        }
                    }
                    ViewerEvent::BondClicked(i) => println!("Main Trace: Bond {} Clicked", i),
                    ViewerEvent::NothingClicked => println!("Main Trace: Nothing Clicked"),
                }
            }

            updates
        },
        // GUI Handler
        |(viewer, _controllers), ctx, _scene| {
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
