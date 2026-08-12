#!/bin/sh
#CGO_ENABLED=1 GOOS=linux go build -buildvcs=false \
# -ldflags="-s -w -X 'main.DefaultConfigPath=config.yaml' -X 'main.Version=7.2.127' -X 'main.Commit=pei5' -X 'main.BuildDate=20260810'" \
# -o ./llmproxy ./cmd/server/

# linux arm64 交叉编译
# 编译（CC 指定 zig cc 交叉编译到 aarch64-linux-musl）
CGO_ENABLED=1 GOOS=linux GOARCH=arm64 CC="zig cc -target aarch64-linux-musl" \
  go build -buildvcs=false \
    -ldflags="-s -w -X 'main.DefaultConfigPath=config.yaml' -X 'main.Version=7.2.127' -X 'main.Commit=pei5' -X 'main.BuildDate=20260810'" \
    -o ./llmproxy-arm64 ./cmd/server/