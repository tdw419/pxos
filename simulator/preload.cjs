// Expose a minimal, isolated bridge to the renderer if needed later.
// Currently empty for security; keeping the scaffold for future native hooks.
const { contextBridge } = require('electron');

contextBridge.exposeInMainWorld('electronAPI', {
  // add safe APIs here when needed
});
