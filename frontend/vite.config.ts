import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// Standalone Vite config for local development, GitHub, and deployment.
// Figma Make's internal preview plugins are intentionally removed.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { host: 'localhost', port: 5173 },
});
