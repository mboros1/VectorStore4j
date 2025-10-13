package io.github.mboros1.vs4j.core.vectors.memory.layout;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;

import java.nio.file.Path;

public record VectorLayout(Path bundlePath,
                           int dim,
                           Dtype dtype,
                           long shardSizeBytes,
                           int rowBytes,
                           int rowStride,
                           int rowsPerShard) {
    public static final long DEFAULT_SHARD_SIZE_BYTES = 1L << 30;
    // alignment must be a power of 2
    private static final int ALIGN = 64;

    public static VectorLayout of(Path bundlePath, int dim, Dtype dtype) {
        return of(bundlePath, dim, dtype, DEFAULT_SHARD_SIZE_BYTES);
    }

    public static VectorLayout of(Path bundlePath, int dim, Dtype dtype, long shardSizeBytes) {
        if (bundlePath == null) throw new IllegalArgumentException("bundlePath null");
        if (dim <= 0) throw new IllegalArgumentException("dim must be > 0");
        int elemSize = switch (dtype) { case F32 -> 4; case F16 -> 2; };
        int rowBytes = Math.multiplyExact(dim, elemSize);
        int rowStride = Math.toIntExact(alignUp(rowBytes));

        if (shardSizeBytes <= 0)
            throw new IllegalArgumentException("shardSizeBytes must be > 0");

        int rowsPerShard = (int) Math.max(1, shardSizeBytes / rowBytes);
        long shardPayload = (long) rowsPerShard * rowBytes;
        if (shardPayload > Integer.MAX_VALUE)
            throw new IllegalArgumentException(STR."single shard mapping too large for slice indexing: \{shardPayload}");

        return new VectorLayout(bundlePath, dim, dtype, shardSizeBytes, rowBytes, rowStride, rowsPerShard);
    }

    // alignment must be a power of 2 (like 64)
    static long alignUp(long size) {
        int mask = ALIGN - 1;
        return (size + mask) & ~mask;
    }

    public int shardOf(int rowId) {
        return rowId / rowsPerShard;
    }

    public int indexInShard(int rowId) {
        return rowId % rowsPerShard;
    }
}