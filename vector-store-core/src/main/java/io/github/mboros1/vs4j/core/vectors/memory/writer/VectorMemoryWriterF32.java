package io.github.mboros1.vs4j.core.vectors.memory.writer;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;
import io.github.mboros1.vs4j.core.vectors.memory.layout.VectorLayout;

import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

final public class VectorMemoryWriterF32 extends BaseVectorMemoryWriter {

    VectorMemoryWriterF32(Path bundlePath, int dim) {
        super(VectorLayout.of(bundlePath, dim, Dtype.F32));
    }

    VectorMemoryWriterF32(Path bundlePath, int dim, long shardSizeBytes) {
        super(VectorLayout.of(bundlePath, dim, Dtype.F32, shardSizeBytes));
    }

    @Override
    protected RowCursor newRow(int rowId, MemorySegment rowSeg, int rowDim) {
        return new RowCursorF32(rowId, rowSeg, rowDim);
    }
}
