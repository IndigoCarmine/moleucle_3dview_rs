use crate::molecule::Molecule;
use crate::render_state::SharedRenderStates;
use crate::offscreen_renderer::RenderStyle;
use crate::viewer::ColorFn;
use lin_alg::f32::Vec3;

pub struct RenderFrameState<'a> {
    pub molecule: Option<&'a Molecule>,
    /// Identifies the state of everything the built-in molecule geometry is
    /// built from — see [`crate::MoleculeViewer::revision`], which is where the
    /// viewport path gets this value. The renderer keys its geometry cache on
    /// it: an unchanged revision means the cached vertex/instance buffers still
    /// describe this frame and no CPU rebuild or GPU upload is needed.
    ///
    /// [`RenderFrameState::new`] defaults it to a fresh value every call, so a
    /// caller assembling frame state by hand always gets a rebuild. Opt into
    /// caching with [`RenderFrameState::with_geometry_revision`].
    pub geometry_revision: u64,
    pub view_proj: [f32; 16],
    pub camera_position: Option<Vec3>,
    pub fov_y: f32,
    pub camera_right: Vec3,
    pub camera_up: Vec3,
    pub camera_forward: Vec3,
    pub color_fn: ColorFn,
    pub shared_states: Option<&'a SharedRenderStates>,
    pub render_style: RenderStyle,
    pub mesh_resolution: usize,
    pub is_low_mode: bool,
    /// Opacity applied to the whole main molecule (atoms + bonds) in
    /// `0.0..=1.0`. Folded into each geometry color's alpha channel so the
    /// molecule can be faded without changing its `color_fn`. `1.0` is opaque.
    pub molecule_opacity: f32,
    /// Optional per-atom sphere radius override, indexed by atom order. When
    /// present, entry `i` replaces the element-derived radius for atom `i`;
    /// indices past the end fall back to the element default. Lets callers draw
    /// e.g. coarse-grained beads through the built-in molecule pipeline.
    pub atom_radii: Option<&'a [f32]>,
    /// Optional per-atom RGBA color override, indexed by atom order. When
    /// present, entry `i` replaces `color_fn`'s result for atom `i` (indices
    /// past the end fall back to `color_fn`). `molecule_opacity` still applies.
    pub atom_colors: Option<&'a [[f32; 4]]>,
    /// Background the color target is cleared to, as straight (non-premultiplied)
    /// RGBA in `0.0..=1.0`. An alpha of `0.0` leaves the background fully
    /// transparent, which is what image export wants; the interactive view keeps
    /// [`DEFAULT_CLEAR_COLOR`].
    pub clear_color: [f32; 4],
}

/// The viewer's own background — the dark blue the interactive view has always
/// used. `RenderFrameState::new` defaults to it so existing callers are
/// unaffected by `clear_color` being added.
pub const DEFAULT_CLEAR_COLOR: [f32; 4] = [0.08, 0.10, 0.14, 1.0];

/// A revision value that is never equal to any previously issued one, so the
/// renderer's geometry cache always misses.
fn next_uncacheable_revision() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    // Start above any plausible `MoleculeViewer::revision` so the two counters
    // cannot collide in a frame that mixes the two sources.
    static COUNTER: AtomicU64 = AtomicU64::new(1 << 63);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

impl<'a> RenderFrameState<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        molecule: Option<&'a Molecule>,
        view_proj: [f32; 16],
        camera_position: Option<Vec3>,
        fov_y: f32,
        camera_right: Vec3,
        camera_up: Vec3,
        camera_forward: Vec3,
        color_fn: ColorFn,
        shared_states: Option<&'a SharedRenderStates>,
        render_style: RenderStyle,
        mesh_resolution: usize,
        is_low_mode: bool,
        molecule_opacity: f32,
    ) -> Self {
        Self {
            molecule,
            // No caller-supplied revision means we cannot know whether anything
            // changed, so assume it did. A fixed default (0) would make a
            // hand-assembled frame cache-hit forever and never redraw.
            geometry_revision: next_uncacheable_revision(),
            view_proj,
            camera_position,
            fov_y,
            camera_right,
            camera_up,
            camera_forward,
            color_fn,
            shared_states,
            render_style,
            mesh_resolution,
            is_low_mode,
            molecule_opacity,
            atom_radii: None,
            atom_colors: None,
            clear_color: DEFAULT_CLEAR_COLOR,
        }
    }

    /// Opt this frame into the renderer's geometry cache by declaring which
    /// version of the geometry inputs it describes — normally
    /// [`crate::MoleculeViewer::revision`].
    ///
    /// Passing a revision that does not change when the molecule, its
    /// positions, its colors or its per-atom overrides change will leave stale
    /// geometry on screen.
    pub fn with_geometry_revision(mut self, geometry_revision: u64) -> Self {
        self.geometry_revision = geometry_revision;
        self
    }

    /// Attach optional per-atom radius / color overrides (see the field docs).
    pub fn with_atom_attrs(
        mut self,
        atom_radii: Option<&'a [f32]>,
        atom_colors: Option<&'a [[f32; 4]]>,
    ) -> Self {
        self.atom_radii = atom_radii;
        self.atom_colors = atom_colors;
        self
    }

    /// Override the background (see [`RenderFrameState::clear_color`]).
    pub fn with_clear_color(mut self, clear_color: [f32; 4]) -> Self {
        self.clear_color = clear_color;
        self
    }
}
