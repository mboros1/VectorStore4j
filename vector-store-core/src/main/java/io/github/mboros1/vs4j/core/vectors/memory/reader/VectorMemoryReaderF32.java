package io.github.mboros1.vs4j.core.vectors.memory.reader;

import io.github.mboros1.vs4j.core.vectors.memory.layout.VectorLayout;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

final public class VectorMemoryReaderF32 extends BaseVectorMemoryReader {
    private static final ValueLayout.OfFloat H32_LE = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    public VectorMemoryReaderF32(VectorLayout layout, int rowCount) {
        super(layout, rowCount);
    }

    @Override
    public float[] fetchRow(int row) {
        float[] result = new float[dim()];
        copyRow(row, result, 0);
        return result;
    }

    private void copyRow(int rowId, float[] dst, int off) {
        final int d = dim();
        if (off < 0 || off + d > dst.length)
            throw new IllegalArgumentException("dst slice out of bounds");
        var row = rowSlice(rowId);
        for (int i=0; i < dim(); i++) {
            dst[i] = row.get(H32_LE, (long)i*Float.BYTES);
        }
    }
}
