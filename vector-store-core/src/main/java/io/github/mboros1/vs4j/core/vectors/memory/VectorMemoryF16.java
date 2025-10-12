package io.github.mboros1.vs4j.core.vectors.memory;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

final public class VectorMemoryF16 extends BaseVectorMemory {

    VectorMemoryF16(Path bundlePath, int dim) {
        super(bundlePath, dim, Dtype.F16);
    }

    @Override
    protected RowCursor newRow(int rowId, MemorySegment rowSeg, int rowDim) {
        return new RowF16(rowId, rowSeg, rowDim);
    }

}
