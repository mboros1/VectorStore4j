package io.github.mboros1.vs4j.core.vectors.memory.writer;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

final public class VectorMemoryF32 extends BaseVectorMemory {

    VectorMemoryF32(Path bundlePath, int dim) {
        super(bundlePath, dim, Dtype.F32);
    }

    VectorMemoryF32(Path bundlePath, int dim, long shardSizeBytes) {
        super(bundlePath, dim, Dtype.F32, shardSizeBytes);
    }

    @Override
    protected RowCursor newRow(int rowId, MemorySegment rowSeg, int rowDim) {
        return new RowF32(rowId, rowSeg, rowDim);
    }
}
