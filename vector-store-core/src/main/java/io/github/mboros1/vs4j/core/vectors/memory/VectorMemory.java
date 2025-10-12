package io.github.mboros1.vs4j.core.vectors.memory;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;

import java.nio.file.Path;

public interface VectorMemory extends AutoCloseable {
    RowCursor allocRow();
    int dim();
    void close() throws Exception;
    Dtype dtype();

    static VectorMemory create(Path bundlePath, int dim, Dtype dtype) {
        return create(bundlePath, dim, dtype, BaseVectorMemory.DEFAULT_SHARD_SIZE_BYTES);
    }

    static VectorMemory create(Path bundlePath, int dim, Dtype dtype, long shardSizeBytes) {
        if (shardSizeBytes <= 0) throw new IllegalArgumentException("shardSizeBytes must be positive");
        return switch (dtype) {
            case F32 -> new VectorMemoryF32(bundlePath, dim, shardSizeBytes);
            case F16 -> new VectorMemoryF16(bundlePath, dim, shardSizeBytes);
        };
    }
}
