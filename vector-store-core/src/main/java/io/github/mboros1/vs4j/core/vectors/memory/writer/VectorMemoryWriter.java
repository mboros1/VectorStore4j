package io.github.mboros1.vs4j.core.vectors.memory.writer;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;

import java.nio.file.Path;

public interface VectorMemoryWriter extends AutoCloseable {
    RowCursor allocRow(int rowId);
    int dim();
    void close() throws Exception;
    Dtype dtype();

    static VectorMemoryWriter create(Path bundlePath, int dim, Dtype dtype) {
        return create(bundlePath, dim, dtype, BaseVectorMemoryWriter.DEFAULT_SHARD_SIZE_BYTES);
    }

    static VectorMemoryWriter create(Path bundlePath, int dim, Dtype dtype, long shardSizeBytes) {
        if (shardSizeBytes <= 0) throw new IllegalArgumentException("shardSizeBytes must be positive");
        return switch (dtype) {
            case F32 -> new VectorMemoryWriterF32(bundlePath, dim, shardSizeBytes);
            case F16 -> new VectorMemoryWriterF16(bundlePath, dim, shardSizeBytes);
        };
    }
}
