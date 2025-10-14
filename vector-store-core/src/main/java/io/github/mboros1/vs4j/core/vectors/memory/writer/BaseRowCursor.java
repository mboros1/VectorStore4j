package io.github.mboros1.vs4j.core.vectors.memory.writer;

import com.fasterxml.jackson.core.JsonParser;
import io.github.mboros1.vs4j.core.vectors.utilities.JsonVector;
import io.github.mboros1.vs4j.core.vectors.utilities.VectorMath;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.FloatBuffer;

sealed public abstract class BaseRowCursor implements RowCursor permits RowCursorF16, RowCursorF32 {

    private final int rowId;
    protected final MemorySegment rowSeg;
    private final int rowDim;
    private final float[] tlBuffer;

    public BaseRowCursor(int rowId, MemorySegment rowSeg, int rowDim, float[] tlBuffer) {
        if (tlBuffer == null ||tlBuffer.length < rowDim)
            throw new IllegalArgumentException(STR."scratch buffer too small: \{tlBuffer == null ? "null" : tlBuffer.length} < \{rowDim}");
        this.rowId = rowId;
        this.rowSeg = rowSeg;
        this.rowDim = rowDim;
        this.tlBuffer = tlBuffer;
    }

    protected abstract void putAtByteOffset(long byteOffset, float value);

    private void normalizeAndSetBytes() {
        VectorMath.normalize(tlBuffer);
        for (int i = 0; i < rowDim; i++) {
            putAtByteOffset(i, tlBuffer[i]);
        }
    }

    @Override
    public int rowIndex() {
        return rowId;
    }

    @Override
    public void putFromJsonArray(JsonParser jp) throws IOException {
        JsonVector.parseVectorFromJson(jp, tlBuffer);
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
