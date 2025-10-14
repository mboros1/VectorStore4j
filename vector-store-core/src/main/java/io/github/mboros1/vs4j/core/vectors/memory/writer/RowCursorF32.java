package io.github.mboros1.vs4j.core.vectors.memory.writer;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

public final class RowCursorF32 extends BaseRowCursor {
    private static final ValueLayout.OfFloat F32_LE =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    public RowCursorF32(int rowId, MemorySegment rowSeg, int rowDim, float[] tlBuffer) {
        super(rowId, rowSeg, rowDim, tlBuffer);
    }

    @Override
    protected void putValue(int elementIndex, float value) {
        rowSeg.set(F32_LE, (long) elementIndex * Float.BYTES, value);
    }
}
