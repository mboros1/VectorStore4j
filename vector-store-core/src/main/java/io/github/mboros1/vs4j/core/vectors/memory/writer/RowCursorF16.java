package io.github.mboros1.vs4j.core.vectors.memory.writer;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

public final class RowCursorF16 extends BaseRowCursor {
    private static final ValueLayout.OfShort F16_LE =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    public RowCursorF16(int rowId, MemorySegment rowSeg, int rowDim, float[] tlBuffer) {
        super(rowId, rowSeg, rowDim, tlBuffer);
    }

    @Override
    protected void putValue(int elementIndex, float value) {
        short h = Float.floatToFloat16(value);
        rowSeg.set(F16_LE, (long) elementIndex * Short.BYTES, h);
    }
}
