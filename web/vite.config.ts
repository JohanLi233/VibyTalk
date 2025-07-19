// vite.config.ts
import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  plugins: [
    vue(),
    viteStaticCopy({
      targets: [
        {
          // 复制所有 wasm 文件到最终输出根目录，开发/生产均可访问
          src: "node_modules/onnxruntime-web/dist/*.wasm",
          dest: ".",
        },
        {
          // 同时复制对应的 mjs 包装文件
          src: "node_modules/onnxruntime-web/dist/*.mjs",
          dest: ".",
        },
      ],
    }),
  ],
  server: {
    host: "0.0.0.0",
    port: 5173,
  },
  worker: {
    format: 'es'
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          'onnxruntime': ['onnxruntime-web'],
          'jszip': ['jszip'],
          'vue-runtime': ['vue']
        }
      }
    },
    chunkSizeWarningLimit: 1000
  }
});
