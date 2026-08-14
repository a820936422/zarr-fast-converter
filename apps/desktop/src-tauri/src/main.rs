// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[cfg(target_os = "linux")]
fn configure_display_backend() {
    let session = std::env::var("XDG_SESSION_TYPE").unwrap_or_default();
    let has_x11_display = std::env::var_os("DISPLAY").is_some();
    let requested = std::env::var("FAST_NC_ZARR_DISPLAY_BACKEND").ok();
    let backend = requested
        .as_deref()
        .or_else(|| (session == "wayland" && has_x11_display).then_some("x11"));

    match backend {
        Some("x11") => {
            std::env::set_var("GDK_BACKEND", "x11");
            std::env::set_var("WINIT_UNIX_BACKEND", "x11");
            std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
        }
        Some("wayland") => {
            std::env::set_var("GDK_BACKEND", "wayland");
            std::env::set_var("WINIT_UNIX_BACKEND", "wayland");
        }
        _ => {}
    }
}

#[cfg(not(target_os = "linux"))]
fn configure_display_backend() {}

fn main() {
    configure_display_backend();
    app_lib::run();
}
