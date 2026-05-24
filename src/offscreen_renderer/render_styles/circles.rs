use lin_alg::f32::Vec3;

use crate::atom_radii::vdw_radius;
use crate::viewer::ColorFn;
use crate::Molecule;

use super::super::MAX_RENDER_VERTICES;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct CircleInstance {
    pub(crate) center: [f32; 3],
    pub(crate) radius: f32,
    pub(crate) color: [f32; 3],
    pub(crate) _pad: f32,
}

pub(crate) fn build_circle_instances(
    molecule: Option<&Molecule>,
    color_fn: ColorFn,
    _camera_position: Option<Vec3>,
) -> Vec<CircleInstance> {
    let Some(mol) = molecule else {
        return Vec::new();
    };

    let mut instances = Vec::with_capacity(mol.atoms.len().min(MAX_RENDER_VERTICES / 6));
    let max_instances = MAX_RENDER_VERTICES / 6;

    for atom in &mol.atoms {
        if instances.len() >= max_instances {
            break;
        }

        let color_tuple = color_fn(atom, false);
        let radius = vdw_radius(&atom.element); // Scale down for better visibility in circles style

        instances.push(CircleInstance {
            center: [atom.position.x, atom.position.y, atom.position.z],
            radius: radius * 0.5, // Scale down for better visibility in circles style
            color: [color_tuple.0, color_tuple.1, color_tuple.2],
            _pad: 0.0,
        });
    }

    instances
}
