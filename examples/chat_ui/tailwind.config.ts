import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "hsl(222 18% 8%)",
        panel: "hsl(222 17% 11%)",
        panel2: "hsl(222 16% 14%)",
        border: "hsl(222 13% 22%)",
        muted: "hsl(218 11% 65%)",
        foreground: "hsl(210 20% 98%)",
        accent: "hsl(154 62% 56%)",
        accent2: "hsl(252 92% 72%)",
      },
      boxShadow: {
        glow: "0 0 60px rgb(75 221 159 / 0.16)",
      },
    },
  },
  plugins: [require("@tailwindcss/typography")],
};

export default config;