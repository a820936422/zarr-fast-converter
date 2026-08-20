mod commands;
mod error;
mod native;
mod pipeline;
mod protocol;
mod resource;
mod tasks;
mod worker;
use serde::Serialize;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
struct BackendInfo {
    app: &'static str,
    version: &'static str,
    runtime: &'static str,
}

#[tauri::command]
fn get_backend_info() -> BackendInfo {
    BackendInfo {
        app: "fast-nc-zarr",
        version: "1.8.1",
        runtime: "tauri-rust",
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(commands::AppState::default())
        .manage(tasks::TaskRegistry::default())
        .invoke_handler(tauri::generate_handler![
            get_backend_info,
            commands::native_capabilities,
            commands::inspect_source,
            commands::inspect_zarr,
            commands::inspect_time_metadata,
            commands::save_inspection_snapshot,
            tasks::get_task,
            tasks::list_tasks,
            tasks::clear_task_history,
            tasks::cancel_task,
            native::start_native_task,
            pipeline::preview_pipeline,
            pipeline::start_pipeline,
            pipeline::start_inspection,
            pipeline::resume_pipeline,
            pipeline::inspect_pipeline_recovery,
        ])
        .setup(|app| {
            app.handle().plugin(tauri_plugin_dialog::init())?;
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
