package io.github.mboros1.vs4j.core.vectors.memory;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

public class VectorMemoryF32 extends BaseVectorMemory {

    VectorMemoryF32(Path bundlePath, int rowDim) {
        super(bundlePath, rowDim, Dtype.F32);
    }

    @Override
    protected RowCursor newRow(int rowId, MemorySegment rowSeg, int rowDim) {
        return new RowF32(rowId, rowSeg, rowDim);
    }
}
