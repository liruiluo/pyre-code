# --- Stage 1: Install dependencies ---
FROM node:18-alpine AS deps
WORKDIR /app
COPY web/package.json web/package-lock.json ./
# The Aliyun host reaches registry.npmjs.org at dial-up speeds; npmmirror
# (also run by Alibaba) is fast from there and serves the same packages.
RUN npm ci --registry=https://registry.npmmirror.com

# --- Stage 2: Build ---
FROM node:18-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY web/ ./
ARG NEXT_PUBLIC_BASE_PATH=""
ENV NEXT_PUBLIC_BASE_PATH=${NEXT_PUBLIC_BASE_PATH}
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm run build

# --- Stage 3: Production ---
FROM node:18-alpine
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs && adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
