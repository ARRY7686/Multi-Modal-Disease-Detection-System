/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        // Dark chrome (dataviz reference palette, dark mode)
        page: '#0d0d0d',
        surface: { DEFAULT: '#1a1a19', 2: '#222220', 3: '#2c2c2a' },
        line: { DEFAULT: '#2c2c2a', strong: '#383835' },
        ink: { DEFAULT: '#f0efec', 2: '#c3c2b7', muted: '#898781' },
        accent: { DEFAULT: '#3987e5', hover: '#5598e7', soft: 'rgba(57,135,229,0.14)' },
        // One fixed colour per prediction mode (categorical slots 1-3)
        mode: { multimodal: '#3987e5', image: '#d95926', text: '#199e70' },
        status: { good: '#0ca30c', warning: '#fab219', serious: '#ec835a', critical: '#d03b3b' },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
      },
    },
  },
  plugins: [],
}
