/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        page: '#0c0b12',
        surface: { DEFAULT: '#17151f', 2: '#1e1b28', 3: '#26222f' },
        line: { DEFAULT: 'rgba(255,255,255,0.08)', strong: 'rgba(255,255,255,0.16)' },
        ink: { DEFAULT: '#f3f1f8', 2: '#bdb8cc', muted: '#7f7a92' },
        violet: { DEFAULT: '#a99ff0', strong: '#9085e9', soft: 'rgba(144,133,233,0.14)' },
        magenta: '#d946ef',
        // One fixed colour per prediction mode (validated as a set on #17151f)
        mode: { multimodal: '#9085e9', image: '#d95926', text: '#199e70' },
        status: { good: '#0ca30c', warning: '#fab219', serious: '#ec835a', critical: '#d03b3b' },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
      },
      keyframes: {
        spin_slow: { to: { transform: 'rotate(360deg)' } },
        scan: { '0%': { top: '-12%' }, '100%': { top: '104%' } },
        ping_soft: { '0%': { transform: 'scale(1)', opacity: '0.7' }, '100%': { transform: 'scale(2.6)', opacity: '0' } },
      },
      animation: {
        'spin-slow': 'spin_slow 60s linear infinite',
        'spin-slower': 'spin_slow 120s linear infinite reverse',
        scan: 'scan 1.8s ease-in-out infinite',
        'ping-soft': 'ping_soft 2s cubic-bezier(0,0,0.2,1) infinite',
      },
    },
  },
  plugins: [],
}
