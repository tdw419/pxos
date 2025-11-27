import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    const lmStudioBase =
      env.VITE_LM_STUDIO_BASE_URL ||
      env.LM_STUDIO_BASE_URL ||
      'http://localhost:1234';

    const normalizedLmBase = lmStudioBase.replace(/\/$/, '');

    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        proxy: {
          // Dev-only proxy to avoid LM Studio CORS errors. Client can call /lmstudio/*.
          '/lmstudio': {
            target: normalizedLmBase,
            changeOrigin: true,
            rewrite: path => path.replace(/^\/lmstudio/, ''),
            secure: false
          }
        }
      },
      plugins: [react()],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
