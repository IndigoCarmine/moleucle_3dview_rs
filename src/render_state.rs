use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// One stored state: the value, and a stamp that changes whenever it is
/// replaced. The stamp is what lets an overlay tell the renderer "nothing I draw
/// has changed" without re-deriving its geometry to find out.
type StateSlot = (u64, Box<dyn Any + Send + Sync>);

/// A map of per-renderer states keyed by type. Use `TypeId::of::<T>()` to store/retrieve `T`.
pub type SharedRenderStates = Arc<Mutex<HashMap<TypeId, StateSlot>>>;

pub fn new_shared_states() -> SharedRenderStates {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Stamp handed to the next stored state. Global rather than per-type so a
/// single counter orders every write, and never repeats.
fn next_state_seq() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(1);
    COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Set state by type. This stores the value under the `TypeId` of `T`.
pub fn set_state_by_type<T: Any + Send + Sync>(states: &SharedRenderStates, value: T) {
    if let Ok(mut map) = states.lock() {
        map.insert(TypeId::of::<T>(), (next_state_seq(), Box::new(value)));
    }
}

/// How many times the state for `T` has been replaced, as an ever-increasing
/// stamp, or `None` if none is stored.
///
/// Overlays fold this into [`crate::AdditionalRender::revision`] so the renderer
/// can skip rebuilding geometry that cannot have changed.
pub fn state_seq_by_type<T: Any + Send + Sync>(states: &SharedRenderStates) -> Option<u64> {
    let map = states.lock().ok()?;
    map.get(&TypeId::of::<T>()).map(|(seq, _)| *seq)
}

/// Borrow the value stored for `T` and run `f` on it, returning `f`'s result, or
/// `None` if no `T` is stored.
///
/// Prefer this to [`get_state_clone_by_type`] on the render path: overlays read
/// their state once per frame, and cloning a state that holds a `Vec` of atom
/// indices or positions means copying the whole thing every frame.
///
/// # Deadlock
///
/// `f` runs while the map's lock is held, and the lock is not reentrant, so `f`
/// must not touch `states` again — no nested `with_state_by_type`, no
/// `get_state_clone_by_type`, no `set_state_by_type`. Reading two state types in
/// one overlay means two *sequential* calls, not nested ones. Calling
/// [`crate::AdditionalRender`] helpers from inside `f` is fine: they only touch
/// the scene.
pub fn with_state_by_type<T: Any + Send + Sync, R>(
    states: &SharedRenderStates,
    f: impl FnOnce(&T) -> R,
) -> Option<R> {
    let map = states.lock().ok()?;
    let value = map.get(&TypeId::of::<T>())?.1.downcast_ref::<T>()?;
    Some(f(value))
}

/// Get a cloned value from the map by type.
///
/// Useful when the caller needs to own the value, or to hold it past the lock.
/// On the per-frame render path prefer [`with_state_by_type`], which borrows.
pub fn get_state_clone_by_type<T: Any + Send + Sync + Clone>(
    states: &SharedRenderStates,
) -> Option<T> {
    if let Ok(map) = states.lock() {
        if let Some((_, boxed)) = map.get(&TypeId::of::<T>()) {
            if let Some(val) = boxed.downcast_ref::<T>() {
                return Some(val.clone());
            }
        }
    }
    None
}

/// Ensure an entry exists for `T` (using `init` if absent) and call `f` with a mutable reference.
pub fn with_state_mut_by_type<T: Any + Send + Sync>(
    states: &SharedRenderStates,
    init: T,
    f: impl FnOnce(&mut T),
) {
    if let Ok(mut map) = states.lock() {
        let key = TypeId::of::<T>();
        map.entry(key)
            .or_insert_with(|| (next_state_seq(), Box::new(init)));

        if let Some((seq, boxed)) = map.get_mut(&key) {
            if let Some(val) = boxed.downcast_mut::<T>() {
                f(val);
                // The caller may have mutated it, and we cannot tell -- assume so.
                *seq = next_state_seq();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard for the removed string-keyed `set_state`, which
    /// inserted its *key* under `TypeId::of::<String>()` and discarded the
    /// caller's value — clobbering any `String` stored through the typed API.
    #[test]
    fn string_state_survives_the_typed_round_trip() {
        let states = new_shared_states();
        set_state_by_type(&states, "payload".to_string());

        assert_eq!(
            get_state_clone_by_type::<String>(&states).as_deref(),
            Some("payload")
        );
    }

    #[test]
    fn absent_and_mismatched_types_read_back_as_none() {
        let states = new_shared_states();
        assert_eq!(get_state_clone_by_type::<u32>(&states), None);

        set_state_by_type(&states, 7u32);
        assert_eq!(get_state_clone_by_type::<u32>(&states), Some(7));
        assert_eq!(get_state_clone_by_type::<i32>(&states), None);
    }
}
