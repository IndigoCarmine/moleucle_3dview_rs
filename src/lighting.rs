//! The lighting every lit pipeline shares: mesh atoms and bonds, sphere
//! impostors, instanced bond cylinders, and the triangle and impostor overlays.
//!
//! All lights are fixed in *view space* — x to the right of the screen, y up,
//! z towards the viewer — so they follow the camera like a headlight: whatever
//! side of the model faces the viewer is the side that is lit, however the
//! model or the camera has been turned. Nothing here depends on the world axes.
//!
//! The rig, as a multiplier on the surface colour:
//!
//! * [`AMBIENT`] everywhere;
//! * a key light from above-left of the viewer ([`KEY_DIR`], [`KEY`]), wrapped
//!   by [`KEY_WRAP`] so its terminator falls a little past the silhouette;
//! * a fill from below-right ([`FILL_DIR`], [`FILL`]), half-Lambert
//!   ([`FILL_WRAP`] = 1), so the side the key misses still has a gradient;
//! * a white Blinn–Phong highlight from the key ([`SPECULAR`],
//!   [`SHININESS`]), added on top of the colour.
//!
//! Over the half of a sphere the viewer can see, the colour multiplier runs
//! from about 1.0 (facing the viewer, slightly towards the key) down to about
//! 0.44 (edge-on, on the side away from both lights): spheres and cylinders
//! keep a clear light-to-shade gradient, and nothing visible falls to black.
//!
//! [`wgsl`] emits the same rig as WGSL from these constants, so the shaders
//! and the tests here cannot drift apart.

/// Colour multiplier every visible surface gets whatever its orientation.
pub const AMBIENT: f32 = 0.32;
/// Strength of the key light.
pub const KEY: f32 = 0.55;
/// Direction *towards* the key light in view space (unnormalised): above-left
/// of the viewer and mostly in front.
pub const KEY_DIR: [f32; 3] = [-0.35, 0.45, 1.0];
/// Wrap of the key light's diffuse term: `max((n·l + w) / (1 + w), 0)`.
pub const KEY_WRAP: f32 = 0.3;
/// Strength of the fill light.
pub const FILL: f32 = 0.20;
/// Direction *towards* the fill light in view space (unnormalised): below-right
/// of the viewer, opposite the key across the line of sight.
pub const FILL_DIR: [f32; 3] = [0.6, -0.4, 0.7];
/// Wrap of the fill light (1 = half-Lambert: it reaches every visible normal).
pub const FILL_WRAP: f32 = 1.0;
/// Peak strength of the white key-light highlight.
pub const SPECULAR: f32 = 0.18;
/// Blinn–Phong exponent of the highlight.
pub const SHININESS: f32 = 48.0;

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(v: [f32; 3]) -> [f32; 3] {
    let len = dot(v, v).sqrt();
    if len > 0.0 {
        [v[0] / len, v[1] / len, v[2] / len]
    } else {
        [0.0, 0.0, 1.0]
    }
}

fn wrapped(cos: f32, wrap: f32) -> f32 {
    ((cos + wrap) / (1.0 + wrap)).max(0.0)
}

/// Unit direction towards the key light, in view space.
pub fn key_dir() -> [f32; 3] {
    normalize(KEY_DIR)
}

/// Unit direction towards the fill light, in view space.
pub fn fill_dir() -> [f32; 3] {
    normalize(FILL_DIR)
}

/// Unit half vector between the key light and the viewer (+z), in view space.
pub fn half_dir() -> [f32; 3] {
    let k = key_dir();
    normalize([k[0], k[1], k[2] + 1.0])
}

/// A world-space normal in the camera's view space (x right, y up, z towards
/// the viewer), given the camera's world-space right, up and forward (from the
/// eye into the scene) axes.
pub fn view_normal(normal: [f32; 3], right: [f32; 3], up: [f32; 3], forward: [f32; 3]) -> [f32; 3] {
    normalize([dot(normal, right), dot(normal, up), -dot(normal, forward)])
}

/// What the surface colour is multiplied by for a (unit) view-space normal:
/// ambient plus the key and fill lights.
pub fn diffuse(n: [f32; 3]) -> f32 {
    AMBIENT
        + KEY * wrapped(dot(n, key_dir()), KEY_WRAP)
        + FILL * wrapped(dot(n, fill_dir()), FILL_WRAP)
}

/// The white highlight added on top of the lit colour for a (unit) view-space
/// normal.
pub fn specular(n: [f32; 3]) -> f32 {
    // No highlight on a face turned away from the key light.
    if dot(n, key_dir()) <= 0.0 {
        return 0.0;
    }
    SPECULAR * dot(n, half_dir()).max(0.0).powf(SHININESS)
}

/// The lit colour of a surface of colour `base` with view-space normal `n`.
pub fn shade(base: [f32; 3], n: [f32; 3]) -> [f32; 3] {
    let n = normalize(n);
    let d = diffuse(n);
    let s = specular(n);
    [base[0] * d + s, base[1] * d + s, base[2] * d + s]
}

fn vec3(v: [f32; 3]) -> String {
    format!("vec3<f32>({:?}, {:?}, {:?})", v[0], v[1], v[2])
}

/// The rig as WGSL: `light_view_normal(n_world)` turns a world-space normal
/// into view space (it reads `uniforms.camera_right/up/forward`, which every
/// lit shader declares), and `light_shade(base, n_view)` is [`shade`].
pub(crate) fn wgsl() -> String {
    format!(
        r#"
// Camera-following lighting, generated from `src/lighting.rs`.
const LIGHT_AMBIENT: f32 = {ambient:?};
const LIGHT_KEY: f32 = {key:?};
const LIGHT_KEY_DIR: vec3<f32> = {key_dir};
const LIGHT_KEY_WRAP: f32 = {key_wrap:?};
const LIGHT_FILL: f32 = {fill:?};
const LIGHT_FILL_DIR: vec3<f32> = {fill_dir};
const LIGHT_FILL_WRAP: f32 = {fill_wrap:?};
const LIGHT_HALF_DIR: vec3<f32> = {half_dir};
const LIGHT_SPECULAR: f32 = {specular:?};
const LIGHT_SHININESS: f32 = {shininess:?};

fn light_view_normal(n: vec3<f32>) -> vec3<f32> {{
    return normalize(vec3<f32>(
        dot(n, uniforms.camera_right.xyz),
        dot(n, uniforms.camera_up.xyz),
        -dot(n, uniforms.camera_forward.xyz),
    ));
}}

fn light_wrapped(c: f32, wrap: f32) -> f32 {{
    return max((c + wrap) / (1.0 + wrap), 0.0);
}}

fn light_shade(base: vec3<f32>, n_in: vec3<f32>) -> vec3<f32> {{
    let n = normalize(n_in);
    let key_cos = dot(n, LIGHT_KEY_DIR);
    let d = LIGHT_AMBIENT
        + LIGHT_KEY * light_wrapped(key_cos, LIGHT_KEY_WRAP)
        + LIGHT_FILL * light_wrapped(dot(n, LIGHT_FILL_DIR), LIGHT_FILL_WRAP);
    var s = 0.0;
    if (key_cos > 0.0) {{
        s = LIGHT_SPECULAR * pow(max(dot(n, LIGHT_HALF_DIR), 0.0), LIGHT_SHININESS);
    }}
    return base * d + vec3<f32>(s, s, s);
}}
"#,
        ambient = AMBIENT,
        key = KEY,
        key_dir = vec3(key_dir()),
        key_wrap = KEY_WRAP,
        fill = FILL,
        fill_dir = vec3(fill_dir()),
        fill_wrap = FILL_WRAP,
        half_dir = vec3(half_dir()),
        specular = SPECULAR,
        shininess = SHININESS,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit normals over the half of a sphere that faces the viewer (z ≥ 0),
    /// the silhouette included.
    fn visible_normals() -> Vec<[f32; 3]> {
        let mut out = Vec::new();
        for i in 0..=45 {
            let theta = (i as f32 * 2.0).to_radians();
            for j in 0..180 {
                let phi = (j as f32 * 2.0).to_radians();
                out.push([
                    theta.sin() * phi.cos(),
                    theta.sin() * phi.sin(),
                    theta.cos(),
                ]);
            }
        }
        out
    }

    fn brightest() -> f32 {
        visible_normals()
            .into_iter()
            .map(diffuse)
            .fold(f32::MIN, f32::max)
    }

    fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
        // v' = q v q*, q = (w, x, y, z) unit.
        let [w, x, y, z] = q;
        let t = [
            2.0 * (y * v[2] - z * v[1]),
            2.0 * (z * v[0] - x * v[2]),
            2.0 * (x * v[1] - y * v[0]),
        ];
        [
            v[0] + w * t[0] + (y * t[2] - z * t[1]),
            v[1] + w * t[1] + (z * t[0] - x * t[2]),
            v[2] + w * t[2] + (x * t[1] - y * t[0]),
        ]
    }

    fn unit_quat(axis: [f32; 3], angle: f32) -> [f32; 4] {
        let a = normalize(axis);
        let (s, c) = (angle * 0.5).sin_cos();
        [c, a[0] * s, a[1] * s, a[2] * s]
    }

    #[test]
    fn the_surface_facing_the_camera_is_lit_close_to_the_brightest() {
        let facing = diffuse([0.0, 0.0, 1.0]);
        let max = brightest();
        assert!(facing >= 0.95 * max, "facing {facing} max {max}");
        // …and brighter than anything seen edge-on or at a glancing angle.
        for n in visible_normals() {
            if n[2] < 0.5 {
                assert!(diffuse(n) < facing, "{n:?}: {} ≥ {facing}", diffuse(n));
            }
        }
        // The brightest colour is the surface colour, not a washed-out one.
        assert!((0.95..=1.05).contains(&max), "{max}");
    }

    #[test]
    fn no_visible_surface_is_dark() {
        let max = brightest();
        for n in visible_normals() {
            let d = diffuse(n);
            assert!(d >= 0.40 * max, "{n:?} lit {d}, under 40 % of {max}");
        }
        // Edge-on, on the side facing away from the key light.
        let k = key_dir();
        let away = normalize([-k[0], -k[1], 0.0]);
        assert!(diffuse(away) >= 0.40 * max, "{}", diffuse(away));
    }

    #[test]
    fn spheres_still_look_round() {
        // Shading across a sphere is not flat: the lit side is clearly brighter
        // than the far edge, and the two edges differ (the light has a side).
        let k = key_dir();
        let toward = normalize([k[0], k[1], 0.0]);
        let away = [-toward[0], -toward[1], 0.0];
        let facing = diffuse([0.0, 0.0, 1.0]);
        assert!(facing - diffuse(away) > 0.4, "{facing} {}", diffuse(away));
        assert!(diffuse(toward) - diffuse(away) > 0.15);
        // The highlight sits towards the key light, not in the middle.
        assert!(specular(half_dir()) > 0.9 * SPECULAR);
        assert!(specular([0.0, 0.0, 1.0]) < specular(half_dir()));
        assert_eq!(specular(away), 0.0);
    }

    #[test]
    fn shading_follows_the_camera() {
        // A camera turned anyhow, and a model turned with it: the same surface
        // seen the same way is shaded the same.
        let camera = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]);
        let normals = [
            [0.0, 0.0, -1.0],
            normalize([0.3, -0.5, -0.8]),
            normalize([1.0, 0.2, 0.0]),
            normalize([-0.6, 0.6, -0.5]),
        ];
        for (axis, angle) in [
            ([0.0, 1.0, 0.0], 1.0),
            ([1.0, 1.0, 0.0], 2.5),
            ([0.2, -0.7, 0.4], -1.9),
            ([0.0, 0.0, 1.0], 3.1),
        ] {
            let q = unit_quat(axis, angle);
            let turned = (
                rotate(q, camera.0),
                rotate(q, camera.1),
                rotate(q, camera.2),
            );
            for n in normals {
                let before = shade(
                    [0.5, 0.6, 0.7],
                    view_normal(n, camera.0, camera.1, camera.2),
                );
                let after = shade(
                    [0.5, 0.6, 0.7],
                    view_normal(rotate(q, n), turned.0, turned.1, turned.2),
                );
                for c in 0..3 {
                    assert!(
                        (before[c] - after[c]).abs() < 1e-4,
                        "{n:?} {before:?} {after:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_side_facing_the_camera_is_lit_whichever_way_the_model_turns() {
        // Orbit the camera around a fixed model: whatever normal points at the
        // eye gets the facing brightness, so no view leaves the front dark.
        let facing = diffuse([0.0, 0.0, 1.0]);
        for (axis, angle) in [
            ([0.0, 1.0, 0.0], 0.0),
            ([0.0, 1.0, 0.0], std::f32::consts::PI),
            ([1.0, 0.0, 0.0], 1.3),
            ([0.4, 0.3, -0.9], 2.2),
        ] {
            let q = unit_quat(axis, angle);
            let (right, up, forward) = (
                rotate(q, [1.0, 0.0, 0.0]),
                rotate(q, [0.0, 1.0, 0.0]),
                rotate(q, [0.0, 0.0, 1.0]),
            );
            let towards_eye = [-forward[0], -forward[1], -forward[2]];
            let d = diffuse(view_normal(towards_eye, right, up, forward));
            assert!((d - facing).abs() < 1e-4, "{d} {facing}");
        }
    }

    #[test]
    fn the_wgsl_carries_the_constants() {
        let src = wgsl();
        assert!(src.contains(&format!("LIGHT_AMBIENT: f32 = {AMBIENT:?};")));
        assert!(src.contains(&vec3(key_dir())));
        assert!(src.contains(&vec3(fill_dir())));
    }
}
