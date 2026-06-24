use crate::atom_radii::vdw_radius;
use crate::viewer::ColorFn;
use crate::{Atom, Molecule};

use super::super::MAX_RENDER_VERTICES;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CircleInstance {
    pub(crate) center: [f32; 3],
    pub(crate) radius: f32,
    pub(crate) color: [f32; 3],
    pub(crate) _pad: f32,
}

/// The impostor pipeline draws each atom as a 6-vertex billboard quad, so the
/// instance budget is the vertex budget divided by 6.
pub(crate) const MAX_IMPOSTOR_INSTANCES: usize = MAX_RENDER_VERTICES / 6;

/// Fill `out` with one sphere-impostor instance per atom, using `radius_for` to
/// size each sphere. `out` is cleared first and its capacity is reused, so
/// trajectory playback rebuilds instances without per-frame allocation. Shared
/// by the `Circles` style and the large-molecule auto-fallback.
pub(crate) fn fill_sphere_instances(
    out: &mut Vec<CircleInstance>,
    molecule: Option<&Molecule>,
    color_fn: ColorFn,
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
            color: [color_tuple.0, color_tuple.1, color_tuple.2],
            _pad: 0.0,
        });
    }
}

pub(crate) fn fill_circle_instances(
    out: &mut Vec<CircleInstance>,
    molecule: Option<&Molecule>,
    color_fn: ColorFn,
) {
    // Scale down for better visibility in the circles style.
    fill_sphere_instances(out, molecule, color_fn, |atom| {
        vdw_radius(&atom.element) * 0.5
    })
}
