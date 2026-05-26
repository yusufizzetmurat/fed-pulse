/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Standalone output is required by the production container so the
  // runtime stage can copy `.next/standalone/server.js` without
  // bringing the full node_modules tree along. Local dev still uses
  // `next dev` and is unaffected.
  output: "standalone",
};

module.exports = nextConfig;
