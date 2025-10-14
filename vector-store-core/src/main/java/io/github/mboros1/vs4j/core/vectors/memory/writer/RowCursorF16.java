package io.github.mboros1.vs4j.core.vectors.memory.writer;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import io.github.mboros1.vs4j.core.vectors.utilities.VectorMath;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public final class RowCursorF16 implements RowCursor {
    private static final ValueLayout.OfShort F16_LE =
            ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    private final int rowId;
    private final MemorySegment rowSeg;
    private final int rowDim;
    private final float[] tlBuffer;

    public RowCursorF16(int rowId, MemorySegment rowSeg, int rowDim, float[] tlBuffer) {
        if (tlBuffer == null ||tlBuffer.length < rowDim)
            throw new IllegalArgumentException(STR."scratch buffer too small: \{tlBuffer == null ? "null" : tlBuffer.length} < \{rowDim}");
        this.rowId = rowId;
        this.rowSeg = rowSeg;
        this.rowDim = rowDim;
        this.tlBuffer = tlBuffer;
    }

    private void setBytes(MemorySegment rowSeg, long idx, short value) {
        rowSeg.set(F16_LE,  idx * Short.BYTES, value);
    }

    @Override
    public int rowIndex() {
        return rowId;
    }

    private void normalizeAndSetBytes() {
        VectorMath.normalize(tlBuffer);
        for (int i = 0; i < rowDim; i++) {
            short h = Float.floatToFloat16(tlBuffer[i]);
            setBytes(rowSeg, i, h);
        }
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
                throw new IOException(STR."non-numeric value in embedding: \{jp.currentToken()}");
            }
            tlBuffer[i++] = v;
        }
        if (i != rowDim) throw new IOException(STR."row \{rowId} incomplete: \{i}/\{rowDim}");
        normalizeAndSetBytes();
    }

    @Override
    public void putArray(float[] src, int off) {
        System.arraycopy(src, off, tlBuffer, 0, rowDim);
        normalizeAndSetBytes();
    }

    @Override
    public void putBuffer(FloatBuffer fb) {
        if (fb.remaining() < rowDim) throw new IllegalArgumentException("not enough data");
        fb.get(tlBuffer, 0, rowDim);
        normalizeAndSetBytes();
    }
}
