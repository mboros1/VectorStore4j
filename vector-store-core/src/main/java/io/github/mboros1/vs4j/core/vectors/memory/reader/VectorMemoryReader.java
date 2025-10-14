package io.github.mboros1.vs4j.core.vectors.memory.reader;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;
import io.github.mboros1.vs4j.core.vectors.memory.layout.VectorLayout;

import java.lang.foreign.MemorySegment;

public interface VectorMemoryReader extends AutoCloseable {
    MemorySegment rowSlice(int rowIdx);
    int dim();
    void close() throws Exception;
    Dtype dtype();
    int numDocs();

    static VectorMemoryReader create(VectorLayout layout, int rowCount) {
        return switch (layout.dtype()) {
            case F32 -> new VectorMemoryReaderF32(layout, rowCount);
            case F16 -> new VectorMemoryReaderF16(layout, rowCount);
        };
    }

    float[] fetchRow(int rowIdx);
}
