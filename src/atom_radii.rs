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
///
/// Matching is case-insensitive and allocation-free: this runs once per atom in
/// every geometry rebuild and once per atom per ray pick, so the previous
/// `to_ascii_uppercase()` cost a heap allocation per atom per frame.
#[inline]
pub fn vdw_radius(element: &str) -> f32 {
    let angstrom = match element.trim().as_bytes() {
        [b'H' | b'h'] => 1.20,
        [b'C' | b'c'] => 1.70,
        [b'N' | b'n'] => 1.55,
        [b'O' | b'o'] => 1.52,
        [b'F' | b'f'] => 1.47,
        [b'P' | b'p'] => 1.80,
        [b'S' | b's'] => 1.80,
        [b'I' | b'i'] => 1.98,
        [b'C' | b'c', b'L' | b'l'] => 1.75,
        [b'B' | b'b', b'R' | b'r'] => 1.85,
        _ => 1.70,
    };

    angstrom * ANGSTROM_TO_NM
}

/// Return the smaller radius used for BallStick atom spheres.
#[inline]
pub fn ball_stick_radius(element: &str, selected: bool) -> f32 {
    let scale = if selected {
        BALL_STICK_SELECTED_ATOM_SCALE
    } else {
        BALL_STICK_ATOM_SCALE
    };

    vdw_radius(element) * scale
}

/// Default BallStick bond radius, expressed relative to the visible H atom size.
///
/// This is the radius the ball-and-stick style draws its sticks at, and the one
/// [`crate::MoleculeViewer::pick`] ray-tests bonds against — the two must agree
/// or bonds become clickable somewhere other than where they are drawn.
#[inline]
pub fn default_ball_stick_bond_radius() -> f32 {
    ball_stick_radius("H", false) * DEFAULT_BOND_RADIUS_SCALE
}

#[cfg(test)]
mod tests {
    use super::{ball_stick_radius, default_ball_stick_bond_radius, vdw_radius, ANGSTROM_TO_NM};

    /// Compare against literal expected values rather than against each other:
    /// a lookup that returned the fallback for *every* input would satisfy a
    /// purely relative assertion.
    #[test]
    fn lookup_is_case_insensitive_and_returns_the_tabulated_radius() {
        for (input, angstrom) in [
            ("H", 1.20),
            ("h", 1.20),
            ("C", 1.70),
            ("c", 1.70),
            (" C ", 1.70),
            ("CL", 1.75),
            ("cl", 1.75),
            ("Cl", 1.75),
            ("cL", 1.75),
            ("BR", 1.85),
            ("Br", 1.85),
            ("I", 1.98),
        ] {
            assert!(
                (vdw_radius(input) - angstrom * ANGSTROM_TO_NM).abs() < 1e-6,
                "vdw_radius({input:?}) should be {angstrom} A"
            );
        }
    }

    /// Unknown symbols, and anything that is not a symbol at all, fall back to
    /// carbon rather than panicking or returning zero.
    #[test]
    fn unknown_symbols_fall_back_to_carbon() {
        for input in ["", "   ", "XYZ", "H2O", "Uub", "\u{3042}"] {
            assert!((vdw_radius(input) - vdw_radius("C")).abs() < 1e-6);
        }
    }

    #[test]
    fn derived_radii_stay_ordered() {
        assert!(ball_stick_radius("H", false) < ball_stick_radius("C", false));
        assert!(ball_stick_radius("C", false) < ball_stick_radius("C", true));
        assert!(default_ball_stick_bond_radius() < ball_stick_radius("H", false));
    }
}
