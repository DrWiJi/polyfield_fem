# AffineWidget3D Life Cycle Report

## Overview

The AffineWidget3D is a PyVista 3D transform widget (translate/rotate) attached to the selected mesh actor. It appears when a mesh is selected and disappears when selection is cleared or the mesh is removed.

---

## Call Order & Life Cycle

### 1. Creation (Widget appears)

**Trigger:** User selects a mesh (click in viewport or list).

**Call chain:**
```
selection_changed (mesh_list or viewport pick)
  → _app.set_selection(idx)
  → _app.selection_changed.emit()
  → _on_mesh_selected()
  → _update_viewport_selection()
  → _sync_affine_widget(selected_id)
  → add_affine_transform_widget(actor, ...)
```

**Conditions:** `selected_id` is set, `actor` exists in `_mesh_actor_by_id`, plotter has `add_affine_transform_widget`.

**Pre-creation:** `disable_picking()` is called (widget uses LMB). Actor transform is baked into `user_matrix`, Position/Orientation/Scale reset to identity.

---

### 2. Update (Same selection, transform changed)

**Trigger:** User drags widget handles, or mesh editor applies transform.

**Call chain (widget drag):**
```
AffineWidget release_callback
  → _apply_affine_matrix_to_mesh(selected_id, user_matrix)
  → _update_affine_widget_origin()  // move widget to new center
```

**Call chain (mesh editor apply):**
```
_apply_mesh_editor_to_model()
  → mesh_viewport_changed.emit()
  → viewport.refresh_meshes()  // see Removal below
```

**Call chain (_update_viewport_selection with same selection):**
```
_sync_affine_widget(selected_id)
  → if selected_id == _affine_widget_mesh_id: _update_affine_widget_origin(); return
```

---

### 3. Removal (Widget disappears)

**Trigger A – Deselection:** User clicks empty space (release without drag).
```
_on_left_release (deselect)
  → _app.set_selection(None)
  → _on_mesh_selected()
  → _update_viewport_selection()
  → _sync_affine_widget(None)
  → _remove_affine_widget()
```

**Trigger B – Mesh removed:** User deletes selected mesh.
```
_action_remove_selected_mesh()
  → _remove_mesh_actor(mesh_id)
  → if mesh_id == _affine_widget_mesh_id: _remove_affine_widget()
  → mesh_viewport_changed.emit()
```

**Trigger C – Viewport refresh (actors replaced):** Mesh list/geometry changes.
```
mesh_viewport_changed.emit()
  → viewport.refresh_meshes()
  → removes ALL mesh actors, creates new ones
  → mesh_actors_updated.emit(new_actor_dict)
  → _on_mesh_actors_updated()
  → _remove_affine_widget()  // FIX: widget was attached to OLD (removed) actor
  → _update_viewport_selection()
  → _sync_affine_widget(selected_id)  // recreates widget with NEW actor
```

**Trigger D – Project load / viewport closed:**
```
_rebuild_viewport_from_project() → _remove_affine_widget()
_restore_viewport_handlers()     → _remove_affine_widget()
```

---

## Root Cause of Unexpected Disappearance

**Bug:** When `viewport.refresh_meshes()` runs, it removes all mesh actors and creates new ones. The AffineWidget was attached to the **old** actor. The widget was not removed, but its actor was gone. `_sync_affine_widget` saw `selected_id == _affine_widget_mesh_id` and only called `_update_affine_widget_origin()`, which used the **new** actor from `_mesh_actor_by_id` while the widget still referenced the removed actor. Result: orphaned/broken widget, invisible or non-functional.

**Fix:** In `_on_mesh_actors_updated()`, call `_remove_affine_widget()` **before** updating `_mesh_actor_by_id` and `_update_viewport_selection()`. This clears the widget when actors are replaced; `_update_viewport_selection` then recreates it for the new actor.

---

## Signal Flow Summary

| Signal                  | Source                    | Effect                                      |
|-------------------------|---------------------------|---------------------------------------------|
| `selection_changed`     | AppModel                  | → `_on_mesh_selected` → `_update_viewport_selection` |
| `mesh_viewport_changed` | MainWindow                | → viewport `refresh_meshes` → actors replaced |
| `mesh_actors_updated`  | Viewport                  | → `_on_mesh_actors_updated` → remove widget, update selection |
| `transform_changed`     | AppModel (widget drag)    | → `_on_transform_changed` (mesh editor only) |

---

## Correct Call Order (after fix)

1. **On mesh_viewport_changed:** Viewport refreshes → removes actors → creates new actors → emits `mesh_actors_updated`.
2. **On mesh_actors_updated:** Remove widget (old actor gone) → update `_mesh_actor_by_id` → `_update_viewport_selection`.
3. **In _update_viewport_selection:** `_sync_affine_widget(selected_id)` → widget is None → create new widget with new actor.
