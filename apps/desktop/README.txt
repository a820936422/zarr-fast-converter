# Fast NC Zarr desktop application

This directory contains the current Tauri 2 + React 19 + TypeScript desktop application.

## Development

From the repository root:

```bash
nvm use
npm --prefix apps/desktop ci
pixi run gui
```

Direct Tauri command:

```bash
npm --prefix apps/desktop run tauri:dev
```

Browser-only preview:

```bash
npm --prefix apps/desktop run dev
```

The browser preview renders the TypeScript UI but does not provide Tauri commands. File inspection and task execution must be verified in the Tauri desktop window.

## Validation

```bash
npm --prefix apps/desktop run typecheck
npm --prefix apps/desktop run build
cargo test -p fast-nc-zarr-desktop
```

The Rust runtime exposes typed commands for inspection, native capabilities, pipeline planning, task events, cancellation and checkpoint recovery. Data-processing compatibility services remain behind those commands and are not part of the frontend API.
