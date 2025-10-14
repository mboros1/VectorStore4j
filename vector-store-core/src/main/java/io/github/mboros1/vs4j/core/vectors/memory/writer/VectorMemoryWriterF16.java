package io.github.mboros1.vs4j.core.vectors.memory.writer;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;
import io.github.mboros1.vs4j.core.vectors.memory.layout.VectorLayout;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

final public class VectorMemoryWriterF16 extends BaseVectorMemoryWriter {

    VectorMemoryWriterF16(Path bundlePath, int dim) {
        super(VectorLayout.of(bundlePath, dim, Dtype.F16));
    }

    VectorMemoryWriterF16(Path bundlePath, int dim, long shardSizeBytes) {
        super(VectorLayout.of(bundlePath, dim, Dtype.F16, shardSizeBytes));
    }

    @Override
    protected RowCursor newRow(int rowId, MemorySegment rowSeg, int rowDim, float[] tlBuffer) {
        return new RowCursorF16(rowId, rowSeg, rowDim, tlBuffer);
    }

}
