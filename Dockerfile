FROM debian:trixie-slim AS build

RUN apt-get update && apt-get install -y \
    gcc \
    make \
    libc6-dev \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Makefile .
COPY src/ ./src/
COPY resources/ ./resources/

RUN make && strip fraude-api


FROM debian:trixie-slim

RUN apt-get update && apt-get install -y \
    libgomp1 \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV OMP_NUM_THREADS=1

COPY --from=build /app/fraude-api .
COPY --from=build /app/resources ./resources

EXPOSE 9999

CMD ["./fraude-api"]