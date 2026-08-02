//! Element radius lookup shared by bond inference and rendering.
//!
//! Radii are quoted in Ångström in the table below (that is how they are
//! tabulated in the literature) and converted to the crate's nanometer
//! convention on the way out.

use crate::ANGSTROM_TO_NM;

pub const BALL_STICK_ATOM_SCALE: f32 = 0.3;
pub const BALL_STICK_SELECTED_ATOM_SCALE: f32 = 0.5;
pub const DEFAULT_BOND_RADIUS_SCALE: f32 = 0.5;

/// Return an approximate van der Waals radius in nanometers.
pub fn vdw_radius(element: &str) -> f32 {
    match element.trim().to_ascii_uppercase().as_str() {
        "H" => 1.20 * ANGSTROM_TO_NM,
        "C" => 1.70 * ANGSTROM_TO_NM,
        "N" => 1.55 * ANGSTROM_TO_NM,
        "O" => 1.52 * ANGSTROM_TO_NM,
        "F" => 1.47 * ANGSTROM_TO_NM,
        "P" => 1.80 * ANGSTROM_TO_NM,
        "S" => 1.80 * ANGSTROM_TO_NM,
        "CL" => 1.75 * ANGSTROM_TO_NM,
        "BR" => 1.85 * ANGSTROM_TO_NM,
        "I" => 1.98 * ANGSTROM_TO_NM,
        _ => 1.70 * ANGSTROM_TO_NM,
    }
}

/// Return the smaller radius used for BallStick atom spheres.
pub fn ball_stick_radius(element: &str, selected: bool) -> f32 {
    let scale = if selected {
        BALL_STICK_SELECTED_ATOM_SCALE
    } else {
        BALL_STICK_ATOM_SCALE
    };

    vdw_radius(element) * scale
}

/// Default BallStick bond radius, expressed relative to the visible H atom size.
pub fn default_ball_stick_bond_radius() -> f32 {
    ball_stick_radius("H", false) * DEFAULT_BOND_RADIUS_SCALE
}

#[cfg(test)]
mod tests {
    use super::{ball_stick_radius, default_ball_stick_bond_radius, vdw_radius};

    #[test]
    fn lookup_is_case_insensitive_for_common_elements() {
        assert!((vdw_radius("c") - vdw_radius("C")).abs() < f32::EPSILON);
        assert!((vdw_radius("cl") - vdw_radius("CL")).abs() < f32::EPSILON);
        assert!(ball_stick_radius("H", false) < ball_stick_radius("C", false));
        assert!(default_ball_stick_bond_radius() < ball_stick_radius("H", false));
    }
}
