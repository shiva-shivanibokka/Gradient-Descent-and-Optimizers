import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Fully client-side app → export a static site (deployable to any static host).
  output: "export",
};

export default nextConfig;
