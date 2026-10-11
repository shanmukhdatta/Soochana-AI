import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

declare const process: any;

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '');
  const groqKey = env.VITE_GROQ_API_KEY || env.GROQ_API_KEY || process.env.VITE_GROQ_API_KEY || process.env.GROQ_API_KEY || '';

  return {
    plugins: [react()],
    define: {
      'import.meta.env.VITE_GROQ_API_KEY': JSON.stringify(groqKey),
      'import.meta.env.GROQ_API_KEY': JSON.stringify(groqKey),
    },
    server: {
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
          rewrite: p => p.replace(/^\/api/, '')
        }
      }
    }
  };
});
