package io.github.mboros1.vs4j.core.vectors.memory.writer;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import io.github.mboros1.vs4j.core.vectors.utilities.JsonVector;
import io.github.mboros1.vs4j.core.vectors.utilities.VectorMath;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public final class RowCursorF16 extends BaseRowCursor {
    private static final ValueLayout.OfShort F16_LE =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    public RowCursorF16(int rowId, MemorySegment rowSeg, int rowDim, float[] tlBuffer) {
        super(rowId, rowSeg, rowDim, tlBuffer);
    }

    @Override
    protected void putAtByteOffset(long byteOffset, float value) {
        short h = Float.floatToFloat16(value);
        rowSeg.set(F16_LE, byteOffset, h);
    }

    void setBytes(MemorySegment rowSeg, long idx, short value) {
        rowSeg.set(F16_LE,  idx * Short.BYTES, value);
    }
}
