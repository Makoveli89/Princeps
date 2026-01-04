/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        gothic: {
          900: '#0a0a0f', // Deepest background
          800: '#12121a', // Card background
          700: '#1c1c2e', // Border/Hover
          gold: '#c5a059', // Accent gold
          purple: '#6d28d9', // Accent purple
          text: '#e2e8f0', // Primary text
          muted: '#94a3b8', // Secondary text
        }
      },
      fontFamily: {
        serif: ['"Playfair Display"', 'serif'],
        sans: ['"Inter"', 'sans-serif'],
        mono: ['"Fira Code"', 'monospace'],
      },
    },
  },
  plugins: [],
}
