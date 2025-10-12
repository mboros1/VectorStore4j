package io.github.mboros1.vs4j.core.vectors.memory;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;

import java.nio.file.Path;

public interface VectorMemory extends AutoCloseable {
    RowCursor allocRow();        // exclusive window for exactly dim elements
    int dim();
    void close() throws Exception;
    default VectorMemory create(Path bundlePath, int dim, Dtype dtype) {
        if (dtype == Dtype.F32) {
            return new VectorMemoryF32(bundlePath, dim);
        } else if (dtype == Dtype.F16) {
            return new VectorMemoryF16(bundlePath, dim);
        }
        throw new IllegalArgumentException("Unsupported dtype: " + dtype);
    }
}
