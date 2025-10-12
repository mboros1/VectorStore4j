package io.github.mboros1.vs4j.core.vectors.memory;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public final class RowF32 implements RowCursor {
    private static final VarHandle F32_LE =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN).varHandle();

    private final int rowId;
    private final MemorySegment rowSeg;
    private final int rowDim;

    public RowF32(int rowId, MemorySegment rowSeg, int rowDim) {
        this.rowId = rowId;
        this.rowSeg = rowSeg;
        this.rowDim = rowDim;
    }

    @Override
    public int rowIndex() {
        return rowId;
    }

    @Override
    public void putFromJsonArray(JsonParser jp) throws IOException {
        if (jp.currentToken() != JsonToken.START_ARRAY)
            throw new IOException("expected START_ARRAY");
        int i = 0;
        while (jp.nextToken() != JsonToken.END_ARRAY) {
            if (i >= rowDim) throw new IOException("too many elements");
            float v = jp.currentToken().isNumeric()
                    ? jp.getFloatValue()
                    : Float.parseFloat(jp.getValueAsString());
            F32_LE.set(rowSeg, (long) i * Float.BYTES, v);
            i++;
        }
        if (i != rowDim) throw new IOException("row " + rowId + " incomplete: " + i + "/" + rowDim);
    }

    @Override
    public void putArray(float[] src, int off) {
        for (int i = 0; i < rowDim; i++) {
            F32_LE.set(rowSeg, (long) i * Float.BYTES, src[off + i]);
        }
    }

    @Override
    public void putBuffer(FloatBuffer fb) {
        if (fb.remaining() < rowDim) throw new IllegalArgumentException("not enough data");
        // Bulk copy via NIO view; avoid creating extra arrays
        FloatBuffer dst = rowSeg.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer();
        // Copy exactly rowDim elements without altering caller’s fb state:
        FloatBuffer src = fb.duplicate();
        int limit = src.position() + rowDim;
        src.limit(limit);
        dst.put(src);
        fb.position(limit);
    }
}
