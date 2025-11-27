<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# Run and deploy your AI Studio app

This contains everything you need to run your app locally.

View your app in AI Studio: https://ai.studio/apps/drive/1iziQR_h9ZhwLkuvhIDY-5TRpp4gGtsUU

## Run Locally

**Prerequisites:**  Node.js


1. Install dependencies:
   `npm install`
2. Set the `GEMINI_API_KEY` in [.env.local](.env.local) to your Gemini API key
3. Run the app:
   `npm run dev`
4. To use LM Studio instead of Gemini, set `LM_STUDIO_BASE_URL` (e.g., `http://localhost:1234`) and optional `LM_STUDIO_MODEL` (default `default`). If `LM_STUDIO_BASE_URL` is present, the app will route chat to LM Studio (OpenAI-compatible API).
5. To add RAG context from LanceDB, run the RAG microservice (`python runtime/rag_http.py`) and set `VITE_RAG_ENDPOINT` (e.g., `http://127.0.0.1:8001/rag`). Optional: `RAG_AGENT_ID` to scope queries (defaults to 0). The service reads `/home/jericho/zion/projects/viber/viber12/lance_data/agent_discussions.lance` by default.

## GPU acceleration

The VRAM visualizer now prefers WebGPU for test patterns/noise/buffer swaps. Modern Chromium/Edge/Arc with the `--enable-unsafe-webgpu` flag (or Chrome 121+) will use your GPU automatically; older browsers will fall back to the CPU simulation. If WebGPU is unavailable you will see `CPU_FALLBACK` in the top bar. No extra config is required beyond running in a WebGPU-capable browser.

## Electron desktop shell

To run outside the browser with your local GPU/adapter:

1. Install dependencies: `npm install`
2. Start the Electron shell (dev server + Electron): `npm run dev:electron`
   - This runs Vite on http://localhost:5173 and opens Electron pointed at it (env vars are set automatically).
   - DevTools auto-open; close them if you prefer.
3. For a built run with a local server: `npm run preview:electron` (runs `vite preview` on 4173 and points Electron at it). If you still prefer file:// load, WebGPU may stay blocked; using the preview server keeps a secure-context URL.

The Electron entry point is `electron-main.cjs`; preload is `preload.cjs` (currently minimal and sandboxed).

### Sandbox note (Linux)
If Electron aborts with a SUID sandbox error, either:
- Run with sandbox disabled (already done in `npm run dev:electron` via `ELECTRON_DISABLE_SANDBOX=1 electron --no-sandbox .`), or
- Fix permissions: `sudo chown root:root node_modules/electron/dist/chrome-sandbox && sudo chmod 4755 node_modules/electron/dist/chrome-sandbox`

WebGPU is enabled in Electron via `app.commandLine.appendSwitch('enable-unsafe-webgpu')` so `navigator.gpu` is available in the desktop shell.
Linux drivers can be picky: the Electron flags also enable `Vulkan` and `SkiaGraphite` features and force ANGLE to `gl`. If you need Vulkan instead, edit `electron-main.cjs` and set `use-angle` to `vulkan`.
