import type { Config } from 'tailwindcss';

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        cellarNoir: '#08090A',
        velvetOlive: '#0F110E',
        champagne: '#F3EED9',
        vintageGold: '#D4AF37',
        goldHoney: '#E5B869',
        crimsonVermillion: '#C83E34',
        mutedSage: '#6E8472',
      },
      fontFamily: {
        serif: ['"Playfair Display"', '"Cormorant Garamond"', 'Georgia', 'serif'],
        sans: ['"Plus Jakarta Sans"', 'Inter', 'system-ui', 'sans-serif'],
      },
      backdropBlur: {
        '2xl': '40px',
      },
    },
  },
  plugins: [],
} satisfies Config;
