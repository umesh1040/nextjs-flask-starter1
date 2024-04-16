/** @type {import('next').NextConfig} */
const nextConfig = {
  rewrites: async () => {
    return [
      {
        source: '/api/:path*',
        destination:
          process.env.NODE_ENV === 'development'
            ? 'https://nextjs-flask-starter1-ten.vercel.app/api/:path*'
            : '/api/',
      },
    ]
  },
}

module.exports = nextConfig
