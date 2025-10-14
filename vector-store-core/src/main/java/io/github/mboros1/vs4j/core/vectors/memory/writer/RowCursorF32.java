package io.github.mboros1.vs4j.core.vectors.memory.writer;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public final class RowCursorF32 implements RowCursor {
    private static final ValueLayout.OfFloat F32_LE =
            ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final int rowId;
    private final MemorySegment rowSeg;
    private final int rowDim;
    private final float[] tlBuffer;

    public RowCursorF32(int rowId, MemorySegment rowSeg, int rowDim, float[] tlBuffer) {
        this.rowId = rowId;
        this.rowSeg = rowSeg;
        this.rowDim = rowDim;
        this.tlBuffer = tlBuffer;
    }

    private void setBytes(MemorySegment rowSeg, long offset, float value) {
        rowSeg.set(F32_LE, offset * Float.BYTES, value);
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
            float v;
            if (jp.currentToken().isNumeric()) {
                v = jp.getFloatValue();
            } else if (jp.currentToken() == JsonToken.VALUE_STRING) {
                v = Float.parseFloat(jp.getValueAsString());
            } else {
                throw new IOException("non-numeric value in embedding: " + jp.currentToken());
            }
            setBytes(rowSeg, i, v);
            i++;
        }
        if (i != rowDim) throw new IOException("row " + rowId + " incomplete: " + i + "/" + rowDim);
    }

    @Override
    public void putArray(float[] src, int off) {
        for (int i = 0; i < rowDim; i++) {
            float v = src[off + i];
            setBytes(rowSeg, i, v);
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
