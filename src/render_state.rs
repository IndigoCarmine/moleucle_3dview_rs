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
