# 分子3Dビューア (Rustライブラリ)

Rustで書かれた軽量な3D分子可視化ライブラリです。[graphics](https://crates.io/crates/graphics)クレート（WGPUベース）をレンダリングに、[bio_files](https://crates.io/crates/bio_files)を分子データの解析に使用しています。

このライブラリは`MoleculeViewer`構造体を提供し、Rustアプリケーションに3D分子モデルのレンダリング機能を組み込むことができます。

## 機能

- **3D可視化**: 分子をボール・アンド・スティックモデルとしてレンダリングします。
  - 原子は球（スフィア）として描画されます。
  - 結合は円柱（シリンダー）として描画されます。
- **元素ごとの色分け**: 原子は元素の種類（C, H, O, N, S, P, Clなど）に基づいて色分けされます。
- **カメラ操作**: Arc-ball制御方式を使用したインタラクティブなカメラ操作（基底のグラフィックスエンジンにより提供）。
- **インタラクション**: 原子および結合のピッキング（選択）機能。
- **ファイルフォーマット**: `bio_files`を介した`.mol2`形式の読み込みヘルパー。



## 使用方法

基本的なビューアアプリケーションを作成する方法の例です：

```rust
use graphics::{run, Scene, UiSettings, GraphicsSettings, EngineUpdates, EntityUpdate, ControlScheme};
use lin_alg::f32::Vec3;
use moleucle_3dview_rs::{Molecule, MoleculeViewer};
use std::path::Path;

fn main() {
    // 1. ビューアの初期化
    let mut viewer = MoleculeViewer::new();

    // 2. 分子のロード
    if let Ok(mol) = Molecule::from_mol2(Path::new("Benzene.mol2")) {
        viewer.set_molecule(mol);
    }

    // 3. シーンの初期化
    let mut scene = Scene::default();
    scene.camera.position = Vec3::new(0.0, 0.0, -10.0);
    scene.input_settings.control_scheme = ControlScheme::Arc {
        center: Vec3::new(0.0, 0.0, 0.0),
    };

    // 4. 分子メッシュでシーンを更新
    viewer.update_scene(&mut scene);

    // 5. アプリケーションループの実行
    run(
        viewer,
        scene,
        UiSettings::default(),
        GraphicsSettings::default(),
        // 描画ハンドラ
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
        // ... (その他のハンドラ)
        |_viewer, _event, _scene, _is_synthetic, _dt| EngineUpdates::default(),
        |viewer, event, scene, _dt| { /* ピッキングなどのイベント処理 */ EngineUpdates::default() },
        |viewer, ctx, scene| { /* 追加UIの描画 */ EngineUpdates::default() },
    );
}
```

完全な実行可能な例については、`examples/simple_viewer.rs`を参照してください。

## サンプルの実行

提供されているサンプルを使用してビューアを動作させるには：

1.  プロジェクトのルートに`.mol2`ファイル（例：`Benzene.mol2`）があることを確認してください。
2.  サンプルを実行します：

```bash
cargo run --example simple_viewer
```

## ライセンス

MIT
