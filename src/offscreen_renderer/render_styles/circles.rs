use crate::atom_radii::vdw_radius;
use crate::viewer::ColorFn;
use crate::{Atom, Molecule};

use super::super::SAFE_MAX_VERTEX_BUFFER_BYTES;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CircleInstance {
    pub(crate) center: [f32; 3],
    pub(crate) radius: f32,
    /// RGBA color; the alpha component drives alpha blending in the shader.
    pub(crate) color: [f32; 4],
}

/// Maximum impostor instances per buffer. Each instance is a small packed
/// struct (not 6 mesh vertices), so the budget is the buffer byte limit
/// divided by the instance size — far more than the old vertex/6 cap, which is
/// what lets multi-million-atom systems (e.g. a 5.4M-atom GROMACS frame) render
/// without dropping atoms. The 240 MB budget keeps the buffer under the common
/// 256 MB GPU per-buffer limit.
pub(crate) const MAX_IMPOSTOR_INSTANCES: usize =
    SAFE_MAX_VERTEX_BUFFER_BYTES / std::mem::size_of::<CircleInstance>();

/// Fill `out` with one sphere-impostor instance per atom, using `radius_for` to
/// size each sphere. `out` is cleared first and its capacity is reused, so
/// trajectory playback rebuilds instances without per-frame allocation. Shared
/// by the `Circles` style and the large-molecule auto-fallback.
pub(crate) fn fill_sphere_instances(
    out: &mut Vec<CircleInstance>,
    molecule: Option<&Molecule>,
    color_fn: ColorFn,
    molecule_opacity: f32,
    radius_for: impl Fn(&Atom) -> f32,
) {
    out.clear();
    let Some(mol) = molecule else {
        return;
    };
    out.reserve(mol.atoms.len().min(MAX_IMPOSTOR_INSTANCES));

    for atom in &mol.atoms {
        if out.len() >= MAX_IMPOSTOR_INSTANCES {
            break;
        }

        let color_tuple = color_fn(atom, false);
        out.push(CircleInstance {
            center: [atom.position.x, atom.position.y, atom.position.z],
            radius: radius_for(atom),
            color: [
                color_tuple.0,
                color_tuple.1,
                color_tuple.2,
                color_tuple.3 * molecule_opacity,
            ],
        });
    }
}

pub(crate) fn fill_circle_instances(
    out: &mut Vec<CircleInstance>,
    molecule: Option<&Molecule>,
    color_fn: ColorFn,
    molecule_opacity: f32,
) {
    // Scale down for better visibility in the circles style.
    fill_sphere_instances(out, molecule, color_fn, molecule_opacity, |atom| {
        vdw_radius(&atom.element) * 0.5
    })
}
