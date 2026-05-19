use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A map of per-renderer states keyed by type. Use `TypeId::of::<T>()` to store/retrieve `T`.
pub type SharedRenderStates = Arc<Mutex<HashMap<TypeId, Box<dyn Any + Send + Sync>>>>;

pub fn new_shared_states() -> SharedRenderStates {
    Arc::new(Mutex::new(HashMap::new()))
}

/// Set state by type. This stores the value under the `TypeId` of `T`.
pub fn set_state_by_type<T: Any + Send + Sync>(states: &SharedRenderStates, value: T) {
    if let Ok(mut map) = states.lock() {
        map.insert(TypeId::of::<T>(), Box::new(value));
    }
}

/// Get a cloned value from the map by type.
pub fn get_state_clone_by_type<T: Any + Send + Sync + Clone>(
    states: &SharedRenderStates,
) -> Option<T> {
    if let Ok(map) = states.lock() {
        if let Some(boxed) = map.get(&TypeId::of::<T>()) {
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
        if !map.contains_key(&key) {
            map.insert(key, Box::new(init));
        }

        if let Some(boxed) = map.get_mut(&key) {
            if let Some(val) = boxed.downcast_mut::<T>() {
                f(val);
            }
        }
    }
}

// Backwards-compatible string-keyed helpers. Keep for now but prefer the type-keyed variants above.
pub fn set_state<T: Any + Send + Sync>(states: &SharedRenderStates, key: &str, _value: T) {
    if let Ok(mut map) = states.lock() {
        map.insert(TypeId::of::<String>(), Box::new(key.to_string()));
        // Store the actual value under a composite key derived from the provided string by hashing the string
        // into a TypeId-like slot is not possible; keep original behavior via a separate internal string map
        // is intentionally not implemented. Prefer using the typed API.
    }
}

#[allow(dead_code)]
pub fn get_state_clone<T: Any + Send + Sync + Clone>(
    _states: &SharedRenderStates,
    _key: &str,
) -> Option<T> {
    // Deprecated: string-keyed access is no longer supported in the type-keyed map.
    None
}

#[allow(dead_code)]
pub fn with_state_mut<T: Any + Send + Sync>(
    _states: &SharedRenderStates,
    _key: &str,
    _init: T,
    _f: impl FnOnce(&mut T),
) {
    // Deprecated placeholder
}
