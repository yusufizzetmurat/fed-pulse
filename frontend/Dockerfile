FROM node:20-alpine@sha256:fb4cd12c85ee03686f6af5362a0b0d56d50c58a04632e6c0fb8363f609372293 AS base

WORKDIR /app


FROM base AS deps

COPY package.json package-lock.json* ./

RUN npm ci


FROM deps AS builder

COPY . .

RUN npm run build


FROM base AS runtime

ENV NODE_ENV=production

COPY --from=deps /app/node_modules ./node_modules
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/public ./public
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/next.config.js ./next.config.js

EXPOSE 3000

CMD ["npm", "run", "start"]


FROM deps AS dev

COPY . .

EXPOSE 3000

CMD ["npm", "run", "dev"]
