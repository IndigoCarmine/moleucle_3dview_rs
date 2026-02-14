# Molecule 3D Viewer (Rust Library)

A lightweight 3D molecule visualization library written in Rust. It utilizes the [graphics](https://crates.io/crates/graphics) crate (based on WGPU) for rendering and [bio_files](https://crates.io/crates/bio_files) for parsing molecular data.

This library provides a `MoleculeViewer` struct that can be integrated into your own Rust applications to render 3D molecule models.

## Features

- **3D Visualization**: Renders molecules as ball-and-stick models.
  - Atoms are rendered as spheres.
  - Bonds are rendered as cylinders.
- **Element Coloring**: Atoms are colored based on their element type (C, H, O, N, S, P, Cl, etc.).
- **Camera Controls**: Interactive camera using an Arc-ball control scheme (handled by the underlying graphics engine).
- **Interaction**: picking support for atoms and bonds.
- **File Format Support**: Helper methods to load `.mol2` files via `bio_files`.



## Usage

Here is a basic example of how to use the library to create a viewer application:

```rust
use graphics::{run, Scene, UiSettings, GraphicsSettings, EngineUpdates, EntityUpdate, ControlScheme};
use lin_alg::f32::Vec3;
use moleucle_3dview_rs::{Molecule, MoleculeViewer};
use std::path::Path;

fn main() {
    // 1. Initialize Viewer State
    let mut viewer = MoleculeViewer::new();

    // 2. Load a molecule
    if let Ok(mol) = Molecule::from_mol2(Path::new("Benzene.mol2")) {
        viewer.set_molecule(mol);
    }

    // 3. Initialize Graphics Scene with Defaults
    let mut scene = Scene::default();
    scene.camera.position = Vec3::new(0.0, 0.0, -10.0);
    scene.input_settings.control_scheme = ControlScheme::Arc {
        center: Vec3::new(0.0, 0.0, 0.0),
    };

    // 4. Update the scene with molecule meshes
    viewer.update_scene(&mut scene);

    // 5. Run the application loop
    run(
        viewer,
        scene,
        UiSettings::default(),
        GraphicsSettings::default(),
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
        // ... (other handlers)
        |_viewer, _event, _scene, _is_synthetic, _dt| EngineUpdates::default(),
        |viewer, event, scene, _dt| { /* Handle events like picking here */ EngineUpdates::default() },
        |viewer, ctx, scene| { /* Draw additional UI here */ EngineUpdates::default() },
    );
}
```

For a complete runnable example, see `examples/simple_viewer.rs`.

## Running the Example

To see the viewer in action using the provided example:

1.  Ensure you have a `.mol2` file (e.g., `Benzene.mol2`) in the project root.
2.  Run the example:

```bash
cargo run --example simple_viewer
```

## License

MIT
