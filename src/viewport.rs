use crate::frame_state::RenderFrameState;
use crate::overlays::{SelectedAtomRender, SelectedAtomRenderState};

use crate::render_state::{new_shared_states, set_state_by_type, SharedRenderStates};
use crate::{
    camera, offscreen_renderer::LodSettings, Camera, Molecule, MoleculeViewer, OffscreenRenderer,
    RenderStyle,
};
use eframe::egui::{self, PointerButton, Sense};
use lin_alg::f32::{Mat4, Vec3};

/// Something the user did to the view, drained by the host with
/// [`InteractiveMoleculeViewport::take_events`].
///
/// Only edges are events. Hovering is a *level* — "which atom is under the
/// pointer right now" — and is read with
/// [`InteractiveMoleculeViewport::hovered_atom`] instead, so a host cannot end
/// up showing a stale hover just because nothing new happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ViewPortEvent {
    /// The atom was clicked. Every click is queued, in order.
    Clicked { atom: usize },
}

/// What to render for an off-screen image export.
///
/// The scene, camera and every registered [`crate::AdditionalRender`] are taken
/// as they currently are, so the export matches the interactive view; only the
/// framing, size and background change.
#[derive(Clone, Copy, Debug)]
pub struct ImageExportRequest {
    /// Output size in pixels, before supersampling.
    pub width: u32,
    pub height: u32,
    /// Sub-rectangle of the on-screen view to frame, as `[x0, y0, x1, y1]` in
    /// normalised `0.0..=1.0` coordinates with **y pointing down**, matching
    /// egui's screen rects. `None` frames the whole view.
    ///
    /// The region's *pixel* aspect (measured against the on-screen viewport size,
    /// see [`InteractiveMoleculeViewport::viewport_size`]) should equal
    /// `width / height`, otherwise the result is stretched — the crop only
    /// re-frames the existing projection, it does not re-fit it.
    pub region: Option<[f32; 4]>,
    /// Straight-alpha RGBA background. `[_, _, _, 0.0]` exports a transparent
    /// background.
    pub clear_color: [f32; 4],
    /// Render at this multiple of the requested size and box-filter down. The
    /// pipelines are single-sampled (no MSAA), so this is the only antialiasing
    /// available. `1` disables it; the factor is reduced automatically when the
    /// scaled size would exceed the device's maximum texture dimension.
    pub supersample: u32,
    /// Sphere/cylinder mesh resolution to force for this render, or `None` to
    /// reuse whatever the interactive view is on.
    ///
    /// Worth setting for an export: the interactive view runs a LOD that drops
    /// the resolution while the camera moves, and geometry that looks fine in a
    /// few hundred pixels is visibly faceted once blown up to a couple of
    /// thousand. The previous value is restored afterwards.
    pub mesh_resolution: Option<usize>,
}

impl Default for ImageExportRequest {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            region: None,
            clear_color: crate::frame_state::DEFAULT_CLEAR_COLOR,
            supersample: 2,
            mesh_resolution: Some(32),
        }
    }
}

/// A rendered image: tightly packed, top-down RGBA8 rows with **straight**
/// (non-premultiplied) alpha, ready to hand to a PNG encoder.
pub struct ExportedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

/// Clip-space matrix that re-frames `region` to fill the whole viewport.
///
/// A sub-rectangle can be mapped by an affine transform on NDC, and an affine
/// NDC transform is a clip-space one because `ndc = clip.xy / clip.w`: writing
/// `clip'.x = sx * clip.x + ox * clip.w` (with `w` untouched) yields
/// `ndc'.x = sx * ndc.x + ox`. That works for perspective and orthographic
/// projections alike, so nothing here needs to know which one is in use.
fn region_crop_matrix(region: [f32; 4]) -> Mat4 {
    let [x0, y0, x1, y1] = region;
    // Screen-normalised (y down) to NDC (y up): nx = 2x - 1, ny = 1 - 2y.
    let (nx_min, nx_max) = (2.0 * x0 - 1.0, 2.0 * x1 - 1.0);
    let (ny_min, ny_max) = (1.0 - 2.0 * y1, 1.0 - 2.0 * y0);

    let sx = 2.0 / (nx_max - nx_min);
    let ox = -(nx_max + nx_min) / (nx_max - nx_min);
    let sy = 2.0 / (ny_max - ny_min);
    let oy = -(ny_max + ny_min) / (ny_max - ny_min);

    // Column-major, matching lin_alg's storage.
    Mat4::new([
        sx, 0.0, 0.0, 0.0, //
        0.0, sy, 0.0, 0.0, //
        0.0, 0.0, 1.0, 0.0, //
        ox, oy, 0.0, 1.0,
    ])
}

/// Box-filter `src` down by `factor` in both axes, staying in premultiplied
/// alpha.
///
/// The color target is already premultiplied (see [`unpremultiply`]), and a plain
/// linear average of premultiplied values is the correct filter — that is exactly
/// why compositing pipelines keep premultiplied intermediates. Averaging straight
/// alpha instead would pull each partially covered pixel's RGB toward the
/// background, which reads as a dark halo around the molecule.
fn box_filter_premultiplied(
    src: &[u8],
    width: u32,
    height: u32,
    factor: u32,
) -> (u32, u32, Vec<u8>) {
    if factor <= 1 {
        return (width, height, src.to_vec());
    }
    let (out_w, out_h) = (width / factor, height / factor);
    let mut out = Vec::with_capacity((out_w * out_h * 4) as usize);
    let samples = (factor * factor) as f32;

    for oy in 0..out_h {
        for ox in 0..out_w {
            let mut acc = [0.0f32; 4];
            for dy in 0..factor {
                for dx in 0..factor {
                    let sx = ox * factor + dx;
                    let sy = oy * factor + dy;
                    let i = ((sy * width + sx) * 4) as usize;
                    for c in 0..4 {
                        acc[c] += src[i + c] as f32;
                    }
                }
            }
            for c in 0..4 {
                out.push((acc[c] / samples).round().clamp(0.0, 255.0) as u8);
            }
        }
    }

    (out_w, out_h, out)
}

/// Convert premultiplied RGBA in place to the straight alpha PNG and image
/// editors expect.
///
/// The pipelines blend with [`wgpu::BlendState::ALPHA_BLENDING`], whose color
/// factors are `SrcAlpha`/`OneMinusSrcAlpha` while its alpha factors are
/// `One`/`OneMinusSrcAlpha`. Drawing a fragment of opacity `a` onto a target
/// cleared to `(0, 0, 0, 0)` therefore leaves `rgb = a * color` with `alpha = a`
/// — premultiplied. Writing that out as-is would darken anything translucent, so
/// a molecule faded to 50% would export at half brightness.
///
/// With an opaque background every pixel ends up at `alpha = 1` and this is a
/// no-op, so it is always safe to apply.
fn unpremultiply(rgba: &mut [u8]) {
    for px in rgba.chunks_exact_mut(4) {
        let a = px[3];
        if a == 0 || a == 255 {
            continue;
        }
        for c in 0..3 {
            let straight = (px[c] as f32) * 255.0 / (a as f32);
            px[c] = straight.round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// The camera a fresh viewport starts with.
///
/// The aspect ratio is only a placeholder — `show()` sets the real one from the
/// widget rect every frame — but it must not be left at the camera's own default,
/// because `render_image` can be called before the first `show()` and would then
/// export at the wrong aspect.
fn default_camera() -> camera::OrbitalCamera {
    let mut camera = camera::OrbitalCamera::default();
    camera.set_aspect(800.0 / 600.0);
    camera
}

/// Everything a hover pick's result depends on.
///
/// Picking is a linear ray test over every atom and every bond, and it runs on
/// every frame the pointer is over the viewport — which, during trajectory
/// playback or an animated UI, is continuous. Nothing about the result can
/// change unless the pointer moved, the camera moved, or the geometry changed,
/// so an unchanged key means the previous answer still holds.
#[derive(Clone, Copy, PartialEq)]
struct HoverPickKey {
    pointer: [u32; 2],
    view_proj: [u32; 16],
    /// `MoleculeViewer::revision` — covers replacing the molecule and moving its
    /// atoms alike.
    geometry_revision: u64,
}

pub struct InteractiveMoleculeViewport {
    viewer: MoleculeViewer,
    camera: camera::OrbitalCamera,
    offscreen: OffscreenRenderer,
    shared_states: SharedRenderStates,
    /// Events raised during [`Self::show`], waiting to be drained by the host.
    events: Vec<ViewPortEvent>,
    /// Atom under the pointer as of the last [`Self::show`], or `None` when the
    /// pointer is off the view or over empty space.
    hovered_atom: Option<usize>,
    /// The last hover pick and the inputs it was computed from, so a stationary
    /// pointer over a still scene reuses the answer instead of re-running the
    /// ray test.
    last_hover_pick: Option<(HoverPickKey, Option<usize>)>,
}

impl InteractiveMoleculeViewport {
    pub fn new() -> Self {
        let viewer = MoleculeViewer::new();
        let shared_states = new_shared_states();
        let mut offscreen = OffscreenRenderer::new();
        offscreen.add_additional_render(Box::new(SelectedAtomRender::new()));

        Self {
            viewer,
            camera: default_camera(),
            offscreen,
            shared_states,
            events: Vec::new(),
            hovered_atom: None,
            last_hover_pick: None,
        }
    }

    /// Expose a way to add an AdditionalRender to the internal viewer from callers.
    pub fn add_additional_render_box(&mut self, render: Box<dyn crate::AdditionalRender>) {
        self.offscreen.add_additional_render(render);
    }

    pub fn set_molecule(&mut self, molecule: Molecule) {
        self.viewer.set_molecule(molecule);
    }

    /// Update atom positions in place for trajectory playback (nanometer
    /// units). See [`MoleculeViewer::update_positions`].
    pub fn update_positions(&mut self, positions: &[Vec3]) -> Result<(), String> {
        self.viewer.update_positions(positions)
    }

    /// Update atom positions in place from Ångström coordinates.
    pub fn update_positions_angstrom(&mut self, coords: &[[f32; 3]]) -> Result<(), String> {
        self.viewer.update_positions_angstrom(coords)
    }

    /// Take the events raised since the last call, oldest first.
    ///
    /// Events are queued during [`Self::show`], so drain them *after* calling it
    /// to react on the same frame.
    ///
    /// ```no_run
    /// # use moleucle_3dview_rs::{InteractiveMoleculeViewport, ViewPortEvent};
    /// # fn demo(vp: &mut InteractiveMoleculeViewport, ui: &mut egui::Ui,
    /// #         rs: &egui_wgpu::RenderState) -> Result<(), String> {
    /// vp.show(ui, rs)?;
    /// for event in vp.take_events() {
    ///     match event {
    ///         ViewPortEvent::Clicked { atom } => println!("clicked {atom}"),
    ///     }
    /// }
    /// if let Some(atom) = vp.hovered_atom() {
    ///     println!("hovering {atom}");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn take_events(&mut self) -> Vec<ViewPortEvent> {
        std::mem::take(&mut self.events)
    }

    /// The atom under the pointer as of the last [`Self::show`], or `None` when
    /// the pointer is off the view or over empty space.
    ///
    /// This is state, not an event: read it every frame and it is always
    /// current, including when the pointer moves off a molecule.
    pub fn hovered_atom(&self) -> Option<usize> {
        self.hovered_atom
    }

    pub fn selected_atoms(&self) -> Vec<usize> {
        crate::render_state::with_state_by_type::<SelectedAtomRenderState, _>(
            &self.shared_states,
            |state| state.selected_atoms.clone(),
        )
        .unwrap_or_default()
    }
    /// Hide part of the molecule without rebuilding it. See
    /// [`MoleculeViewer::set_visible_atoms`] — indices never renumber, so
    /// picking results and every per-atom array stay in full-molecule space.
    pub fn set_visible_atoms(&mut self, visible: Option<Vec<bool>>) {
        self.viewer.set_visible_atoms(visible);
    }

    /// Frame the camera on what is currently drawn.
    ///
    /// Fits the *visible* atoms: hiding the solvent and then resetting the view
    /// should frame the solute, not the box it used to sit in.
    pub fn focus_on_molecule_center(&mut self) {
        let Some(molecule) = self.viewer.molecule.as_ref() else {
            return;
        };

        let mut sum = Vec3::new(0.0, 0.0, 0.0);
        let mut count = 0usize;
        for (index, atom) in molecule.atoms.iter().enumerate() {
            if self.viewer.is_atom_visible(index) {
                sum = sum + atom.position;
                count += 1;
            }
        }
        if count == 0 {
            return;
        }

        let center = sum / count as f32;
        let mut radius = 0.0_f32;
        for (index, atom) in molecule.atoms.iter().enumerate() {
            if self.viewer.is_atom_visible(index) {
                radius = radius.max((atom.position - center).magnitude());
            }
        }

        self.camera.center = center;
        self.camera.radius = radius.max(1e-3) * 2.0;
    }

    pub fn set_state_by_type<T: 'static + Send + Sync>(&mut self, state: T) {
        set_state_by_type(&self.shared_states, state);
    }

    pub fn render_style(&self) -> RenderStyle {
        self.offscreen.render_style()
    }

    pub fn mesh_resolution(&self) -> usize {
        self.offscreen.mesh_resolution()
    }

    pub fn set_mesh_resolution(&mut self, mesh_resolution: usize) {
        self.offscreen.set_mesh_resolution(mesh_resolution);
    }

    pub fn lod_settings(&self) -> LodSettings {
        self.offscreen.lod_settings()
    }

    pub fn set_lod_settings(&mut self, lod_settings: LodSettings) {
        self.offscreen.set_lod_settings(lod_settings);
    }

    pub fn set_render_style(&mut self, render_style: RenderStyle) {
        self.offscreen.set_render_style(render_style);
    }

    /// Current whole-molecule opacity (`0.0..=1.0`).
    pub fn molecule_opacity(&self) -> f32 {
        self.viewer.molecule_opacity
    }

    /// Set the whole-molecule opacity (atoms + bonds), clamped to `0.0..=1.0`.
    /// `1.0` is fully opaque; lower values fade the main molecule via alpha
    /// blending. The additional-render overlays are unaffected.
    pub fn set_molecule_opacity(&mut self, opacity: f32) {
        self.viewer.set_molecule_opacity(opacity);
    }

    /// Override the per-atom sphere radius of the main molecule (atom order),
    /// or pass `None` to use element-derived radii. Lets callers draw
    /// coarse-grained beads through the built-in pipeline (so they participate
    /// in shading, picking and opacity) instead of a separate overlay.
    pub fn set_atom_radii(&mut self, radii: Option<Vec<f32>>) {
        self.viewer.set_atom_radii(radii);
    }

    /// Override the per-atom RGBA color of the main molecule (atom order), or
    /// pass `None` to use the color function.
    pub fn set_atom_colors(&mut self, colors: Option<Vec<[f32; 4]>>) {
        self.viewer.set_atom_colors(colors);
    }

    pub fn free_egui_texture(&mut self, render_state: &egui_wgpu::RenderState) {
        self.offscreen.free_egui_texture(render_state);
    }

    /// Size in pixels of the color target the interactive view last rendered at,
    /// i.e. the on-screen viewport size. `(0, 0)` before the first
    /// [`Self::show`]. Callers need it to convert a normalised region into a
    /// pixel aspect ratio.
    pub fn viewport_size(&self) -> (u32, u32) {
        self.offscreen.size()
    }

    /// Render the current scene off-screen at an arbitrary size and return the
    /// pixels, for image export.
    ///
    /// The camera, molecule, style and every registered overlay are used exactly
    /// as they stand, so the output matches what is on screen; `request` only
    /// controls the framing, the size and the background. Callers that want an
    /// overlay left out should hide it through its render state first (e.g.
    /// `set_state_by_type(AxisRenderState { visible: false })`) and restore it
    /// afterwards.
    ///
    /// This resizes the shared color target and blocks on a GPU readback, so the
    /// interactive view must be re-rendered afterwards — call it *before*
    /// [`Self::show`] within a frame and `show` will resize it back on its own.
    pub fn render_image(
        &mut self,
        render_state: &egui_wgpu::RenderState,
        request: &ImageExportRequest,
    ) -> Result<ExportedImage, String> {
        if request.width == 0 || request.height == 0 {
            return Err("Export size must be non-zero".to_string());
        }
        let max_dim = render_state.device.limits().max_texture_dimension_2d;
        if request.width > max_dim || request.height > max_dim {
            return Err(format!(
                "Export size {}x{} exceeds this device's maximum texture size of {max_dim}",
                request.width, request.height
            ));
        }

        // Trim the supersample factor to whatever still fits in a texture.
        let mut factor = request.supersample.max(1);
        while factor > 1
            && (request.width * factor > max_dim || request.height * factor > max_dim)
        {
            factor -= 1;
        }
        let (render_w, render_h) = (request.width * factor, request.height * factor);

        let previous_size = self.offscreen.size();
        let previous_mesh = self.offscreen.mesh_resolution();
        if let Some(resolution) = request.mesh_resolution {
            // Lock the LOD first: it runs at the top of the render and would
            // otherwise put its own resolution back before anything is drawn.
            self.offscreen.set_lod_locked(true);
            self.offscreen.set_mesh_resolution(resolution);
        }

        let view_proj = {
            let base = self.camera.view_projection();
            match request.region {
                Some(region) => (region_crop_matrix(region) * base).data,
                None => base.data,
            }
        };
        let cam_rot = self.camera.camera_rotation();
        let camera_right = cam_rot.rotate_vec(Vec3::new(1.0, 0.0, 0.0));
        let camera_up = cam_rot.rotate_vec(Vec3::new(0.0, 1.0, 0.0));
        let camera_forward = cam_rot.rotate_vec(Vec3::new(0.0, 0.0, 1.0));

        self.offscreen
            .ensure_resources(render_state, render_w, render_h)?;

        let frame = RenderFrameState::new(
            self.viewer.molecule.as_ref(),
            view_proj,
            Some(self.camera.position()),
            self.camera.fov_y(),
            camera_right,
            camera_up,
            camera_forward,
            self.viewer.color_fn,
            Some(&self.shared_states),
            self.offscreen.render_style(),
            self.offscreen.mesh_resolution(),
            // Always export at full detail, even if the interactive view has
            // dropped to low mode while the user was dragging.
            false,
            self.viewer.molecule_opacity,
        )
        .with_geometry_revision(self.viewer.revision())
        .with_visible_atoms(self.viewer.visible_atoms())
        .with_atom_attrs(
            self.viewer.atom_radii.as_deref(),
            self.viewer.atom_colors.as_deref(),
        )
        .with_clear_color(request.clear_color);

        let render_result = self
            .offscreen
            .render_frame_with_state(render_state, &frame)
            .and_then(|()| self.offscreen.read_rgba(render_state));

        // Restore everything whatever happened, so a failed export cannot leave
        // the interactive view rendering at export resolution or detail.
        if request.mesh_resolution.is_some() {
            self.offscreen.set_mesh_resolution(previous_mesh);
            self.offscreen.set_lod_locked(false);
        }
        if previous_size.0 > 0 && previous_size.1 > 0 {
            let _ = self
                .offscreen
                .ensure_resources(render_state, previous_size.0, previous_size.1);
        }

        let rgba = render_result?;
        // Filter while still premultiplied, then convert once at the end.
        let (width, height, mut rgba) =
            box_filter_premultiplied(&rgba, render_w, render_h, factor);
        unpremultiply(&mut rgba);
        Ok(ExportedImage {
            width,
            height,
            rgba,
        })
    }

    pub fn show(
        &mut self,
        ui: &mut egui::Ui,
        render_state: &egui_wgpu::RenderState,
    ) -> Result<(), String> {
        let available = ui.available_size_before_wrap();
        let width = available.x.max(1.0) as u32;
        let height = available.y.max(1.0) as u32;
        self.camera
            .set_aspect(width as f32 / height as f32);

        if let Some(molecule) = self.viewer.molecule.as_ref() {
            let camera_position = self.camera.position();
            let distance = (camera_position - molecule.center()).magnitude();
            self.offscreen.submit_lod_distance(distance);
        }

        self.offscreen
            .ensure_resources(render_state, width, height)?;

        let view_proj = self.camera.view_projection().data;
        let cam_rot = self.camera.camera_rotation();
        let camera_right = cam_rot.rotate_vec(Vec3::new(1.0, 0.0, 0.0));
        let camera_up = cam_rot.rotate_vec(Vec3::new(0.0, 1.0, 0.0));
        let camera_forward = cam_rot.rotate_vec(Vec3::new(0.0, 0.0, 1.0));

        let frame = RenderFrameState::new(
            self.viewer.molecule.as_ref(),
            view_proj,
            Some(self.camera.position()),
            self.camera.fov_y(),
            camera_right,
            camera_up,
            camera_forward,
            self.viewer.color_fn,
            Some(&self.shared_states),
            self.offscreen.render_style(),
            self.offscreen.mesh_resolution(),
            self.offscreen.is_low_mode(),
            self.viewer.molecule_opacity,
        )
        .with_geometry_revision(self.viewer.revision())
        .with_visible_atoms(self.viewer.visible_atoms())
        .with_atom_attrs(
            self.viewer.atom_radii.as_deref(),
            self.viewer.atom_colors.as_deref(),
        );

        self.offscreen
            .render_frame_with_state(render_state, &frame)?;

        let texture_id = self
            .offscreen
            .texture_id()
            .ok_or_else(|| "No texture id registered".to_string())?;

        let response = ui.add(
            egui::Image::from_texture(egui::load::SizedTexture::new(
                texture_id,
                egui::vec2(width as f32, height as f32),
            ))
            .sense(Sense::click_and_drag()),
        );

        let ctx = ui.ctx();
        self.handle_interaction(ctx, &response);
        Ok(())
    }

    /// Key describing everything the hover pick at `pointer` depends on.
    fn hover_pick_key(&self, pointer: egui::Pos2) -> HoverPickKey {
        let view_proj = self.camera.view_projection().data;
        let mut view_proj_bits = [0u32; 16];
        for (dst, src) in view_proj_bits.iter_mut().zip(view_proj.iter()) {
            *dst = src.to_bits();
        }

        HoverPickKey {
            pointer: [pointer.x.to_bits(), pointer.y.to_bits()],
            view_proj: view_proj_bits,
            geometry_revision: self.viewer.revision(),
        }
    }

    fn handle_interaction(&mut self, ctx: &egui::Context, response: &egui::Response) {
        // Recomputed below whenever the pointer is over the view; leaving the
        // view therefore clears it.
        self.hovered_atom = None;

        if response.hovered() {
            if let Some(pointer) = response.hover_pos() {
                // Reuse the previous answer while nothing that could change it
                // has moved. Without this, parking the cursor over the view
                // costs a full ray scan of every atom and every bond, every
                // frame. The event still fires every frame, so hosts that
                // consume the hover per-frame see no behaviour change.
                let key = self.hover_pick_key(pointer);
                let picked = match self.last_hover_pick {
                    Some((cached_key, picked)) if cached_key == key => picked,
                    _ => {
                        let local = pointer - response.rect.min;
                        let (ray_origin, ray_dir) = self.camera.ray_from_screen(
                            local.x,
                            local.y,
                            response.rect.width().max(1.0),
                            response.rect.height().max(1.0),
                        );

                        let picked = match self.viewer.pick(ray_origin, ray_dir) {
                            Some(crate::viewer::ViewerEvent::AtomClicked(i)) => Some(i),
                            _ => None,
                        };
                        self.last_hover_pick = Some((key, picked));
                        picked
                    }
                };

                self.hovered_atom = picked;
            }

            let scroll = ctx.input(|i| i.smooth_scroll_delta.y);
            if scroll.abs() > f32::EPSILON {
                self.camera.dolly(scroll * 0.02);
            }
        }

        // Primary drag: orbit (Shift+drag or middle/right drag: pan)
        if response.dragged_by(PointerButton::Primary) {
            let delta = response.drag_delta();
            let shift_down = ctx.input(|i| i.modifiers.shift);
            if shift_down {
                self.camera
                    .pan(lin_alg::f32::Vec2::new(delta.x * 0.01, delta.y * 0.01));
            } else {
                self.camera
                    .orbit(delta.x * 0.005, delta.y * 0.005);
            }
        }

        if response.dragged_by(PointerButton::Secondary)
            || response.dragged_by(PointerButton::Middle)
        {
            let delta = response.drag_delta();
            self.camera
                .pan(lin_alg::f32::Vec2::new(delta.x * 0.01, delta.y * 0.01));
        }

        if response.clicked_by(PointerButton::Primary) {
            if let Some(pointer) = response.interact_pointer_pos() {
                let local = pointer - response.rect.min;
                let (ray_origin, ray_dir) = self.camera.ray_from_screen(
                    local.x,
                    local.y,
                    response.rect.width().max(1.0),
                    response.rect.height().max(1.0),
                );

                if let Some(crate::viewer::ViewerEvent::AtomClicked(atom)) =
                    self.viewer.pick(ray_origin, ray_dir)
                {
                    self.events.push(ViewPortEvent::Clicked { atom });
                }
            }
        }
    }
}

impl Default for InteractiveMoleculeViewport {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lin_alg::f32::Vec4;

    /// Push a clip-space point through a crop matrix and return its NDC.
    fn ndc_after(region: [f32; 4], clip: Vec4) -> (f32, f32) {
        let out = region_crop_matrix(region) * clip;
        (out.x / out.w, out.y / out.w)
    }

    /// A clip point whose NDC is `(x, y)` at `w = 1`.
    fn clip_at(x: f32, y: f32) -> Vec4 {
        Vec4 { x, y, z: 0.5, w: 1.0 }
    }

    #[test]
    fn the_full_region_is_the_identity() {
        for (x, y) in [(-1.0, -1.0), (0.0, 0.0), (1.0, 1.0), (0.3, -0.7)] {
            let (nx, ny) = ndc_after([0.0, 0.0, 1.0, 1.0], clip_at(x, y));
            assert!((nx - x).abs() < 1e-5, "x {nx} != {x}");
            assert!((ny - y).abs() < 1e-5, "y {ny} != {y}");
        }
    }

    #[test]
    fn a_region_maps_its_own_corners_to_the_ndc_corners() {
        // Top-left quadrant in screen coords (y down).
        let region = [0.0, 0.0, 0.5, 0.5];
        // Screen (0,0) is NDC (-1, 1); screen (0.5, 0.5) is NDC (0, 0).
        let (nx, ny) = ndc_after(region, clip_at(-1.0, 1.0));
        assert!((nx + 1.0).abs() < 1e-5 && (ny - 1.0).abs() < 1e-5, "{nx},{ny}");
        let (nx, ny) = ndc_after(region, clip_at(0.0, 0.0));
        assert!((nx - 1.0).abs() < 1e-5 && (ny + 1.0).abs() < 1e-5, "{nx},{ny}");
    }

    #[test]
    fn a_centred_region_keeps_the_centre_and_magnifies() {
        let region = [0.25, 0.25, 0.75, 0.75];
        let (nx, ny) = ndc_after(region, clip_at(0.0, 0.0));
        assert!(nx.abs() < 1e-5 && ny.abs() < 1e-5, "centre stays put");
        // Half-size region doubles the scale.
        let (nx, _) = ndc_after(region, clip_at(0.5, 0.0));
        assert!((nx - 1.0).abs() < 1e-5, "{nx}");
    }

    #[test]
    fn region_y_is_screen_down_not_ndc_up() {
        // The upper half of the screen must keep the upper half of the scene.
        let (_, ny) = ndc_after([0.0, 0.0, 1.0, 0.5], clip_at(0.0, 1.0));
        assert!((ny - 1.0).abs() < 1e-5, "top of screen stays at the top: {ny}");
        let (_, ny) = ndc_after([0.0, 0.0, 1.0, 0.5], clip_at(0.0, 0.0));
        assert!((ny + 1.0).abs() < 1e-5, "screen middle becomes the bottom: {ny}");
    }

    /// The full export tail: filter, then convert to straight alpha.
    fn export_pixels(src: &[u8], width: u32, height: u32, factor: u32) -> (u32, u32, Vec<u8>) {
        let (w, h, mut out) = box_filter_premultiplied(src, width, height, factor);
        unpremultiply(&mut out);
        (w, h, out)
    }

    #[test]
    fn filtering_is_a_no_op_at_factor_one() {
        let src = vec![1, 2, 3, 255, 5, 6, 7, 255];
        let (w, h, out) = export_pixels(&src, 2, 1, 1);
        assert_eq!((w, h), (2, 1));
        assert_eq!(out, src);
    }

    #[test]
    fn filtering_averages_a_uniform_block() {
        // 2x2 of the same opaque colour collapses to that colour.
        let src: Vec<u8> = [[10u8, 20, 30, 255]; 4].concat();
        let (w, h, out) = export_pixels(&src, 2, 2, 2);
        assert_eq!((w, h), (1, 1));
        assert_eq!(out, vec![10, 20, 30, 255]);
    }

    #[test]
    fn a_translucent_pixel_exports_at_full_brightness() {
        // What the GPU leaves for a 50%-opacity red over a transparent clear:
        // premultiplied, so rgb is already halved. Writing that straight out
        // would export the molecule at half brightness — the whole reason
        // `unpremultiply` exists.
        let src = vec![128u8, 0, 0, 128];
        let (_, _, out) = export_pixels(&src, 1, 1, 1);
        assert_eq!(out[0], 255, "red restored to full, not {}", out[0]);
        assert_eq!(out[3], 128, "opacity preserved");
    }

    #[test]
    fn a_transparent_edge_does_not_darken_the_colour() {
        // Two opaque red pixels next to two fully transparent ones. Averaging
        // straight alpha would give a half-brightness red — the dark halo the
        // premultiplied filter avoids. The colour must stay saturated and only
        // the alpha may drop.
        let opaque = [255u8, 0, 0, 255];
        let clear = [0u8, 0, 0, 0];
        let src: Vec<u8> = [opaque, opaque, clear, clear].concat();
        let (_, _, out) = export_pixels(&src, 2, 2, 2);
        assert_eq!(out[0], 255, "red stays fully saturated, not {}", out[0]);
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 0);
        assert_eq!(out[3], 128, "alpha carries the coverage");
    }

    #[test]
    fn a_fully_transparent_block_stays_transparent() {
        let src: Vec<u8> = [[0u8, 0, 0, 0]; 4].concat();
        let (_, _, out) = export_pixels(&src, 2, 2, 2);
        assert_eq!(out, vec![0, 0, 0, 0]);
    }

    #[test]
    fn an_opaque_background_is_untouched_by_the_alpha_conversion() {
        // Every pixel comes back at alpha 1 with an opaque clear, so the
        // conversion must not alter a single channel.
        let mut px = vec![9u8, 200, 77, 255, 0, 1, 2, 255];
        let before = px.clone();
        unpremultiply(&mut px);
        assert_eq!(px, before);
    }

    #[test]
    fn the_default_request_keeps_the_viewer_background() {
        let request = ImageExportRequest::default();
        assert_eq!(request.clear_color, crate::DEFAULT_CLEAR_COLOR);
        assert!(request.region.is_none());
    }
}
