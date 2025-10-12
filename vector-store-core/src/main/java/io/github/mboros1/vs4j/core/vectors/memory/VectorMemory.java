package io.github.mboros1.vs4j.core.vectors.memory;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;

import java.nio.file.Path;

public interface VectorMemory extends AutoCloseable {
    RowCursor allocRow();
    int dim();
    void close() throws Exception;
    Dtype dtype();

    static VectorMemory create(Path bundlePath, int dim, Dtype dtype) {
        return switch (dtype) {
            case F32 -> new VectorMemoryF32(bundlePath, dim);
            case F16 -> new VectorMemoryF16(bundlePath, dim);
        };
    }
}
