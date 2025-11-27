// Electron entry point to wrap the Vite React app in a desktop shell.
// Uses dev server in dev; loads built files in production.
const { app, BrowserWindow, nativeTheme } = require('electron');
const path = require('path');

// Enable WebGPU inside Electron (needed for navigator.gpu).
app.commandLine.appendSwitch('enable-unsafe-webgpu');
// Linux often needs explicit feature enables for GPU backends.
app.commandLine.appendSwitch('enable-features', 'AllowUnsafeWebGPU,Vulkan,SkiaGraphite');
// Try GL/EGL path first for broader driver compatibility; swap to vulkan if needed.
app.commandLine.appendSwitch('use-angle', 'gl');

const devServerURL = process.env.ELECTRON_DEV_SERVER_URL || process.env.VITE_DEV_SERVER_URL;
const isDev = !!devServerURL;
const preloadPath = path.join(__dirname, 'preload.cjs');

function createWindow() {
  const win = new BrowserWindow({
    width: 1200,
    height: 800,
    backgroundColor: '#000000',
    webPreferences: {
      contextIsolation: true,
      preload: preloadPath,
    },
  });

  // Keep dark styling consistent with the app theme.
  nativeTheme.themeSource = 'dark';

  if (isDev) {
    win.loadURL(devServerURL);
    win.webContents.openDevTools({ mode: 'detach' });
  } else {
    const indexPath = path.join(__dirname, 'dist', 'index.html');
    win.loadFile(indexPath);
  }
}

app.whenReady().then(() => {
  createWindow();

  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
