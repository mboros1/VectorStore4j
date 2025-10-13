package io.github.mboros1.vs4j.core.vectors.memory;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.ShortBuffer;

public final class RowF16 implements RowCursor {
    private static final ValueLayout.OfShort F16_LE =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final int rowId;
    private final MemorySegment rowSeg;
    private final int rowDim;

    public RowF16(int rowId, MemorySegment rowSeg, int rowDim) {
        this.rowId = rowId;
        this.rowSeg = rowSeg;
        this.rowDim = rowDim;
    }

    private void setBytes(MemorySegment rowSeg, long offset, short value) {
        rowSeg.set(F16_LE, offset * Short.BYTES, value);
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
            short h = Float.floatToFloat16(v);
            setBytes(rowSeg, i, h);
            i++;
        }
        if (i != rowDim) throw new IOException("row " + rowId + " incomplete: " + i + "/" + rowDim);

    }

    @Override
    public void putArray(float[] src, int off) {
        for (int i = 0; i < rowDim; i++) {
            final short h = Float.floatToFloat16(src[off + i]);
            setBytes(rowSeg, i, h);
        }
    }

    @Override
    public void putBuffer(FloatBuffer fb) {
        if (fb.remaining() < rowDim) throw new IllegalArgumentException("not enough data");
        ShortBuffer dst = rowSeg.asByteBuffer().order(ByteOrder.LITTLE_ENDIAN).asShortBuffer();
        FloatBuffer src = fb.duplicate();
        int start = src.position();
        int limit = start + rowDim;
        src.limit(limit);

        while (src.hasRemaining()) {
            dst.put(Float.floatToFloat16(src.get()));
        }
        fb.position(limit);
    }
}
