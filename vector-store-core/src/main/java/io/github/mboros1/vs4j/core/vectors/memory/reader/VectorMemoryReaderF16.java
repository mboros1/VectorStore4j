package io.github.mboros1.vs4j.core.vectors.memory.reader;

import io.github.mboros1.vs4j.core.vectors.memory.layout.VectorLayout;

import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

final public class VectorMemoryReaderF16 extends BaseVectorMemoryReader {
    private static final ValueLayout.OfShort H16_LE = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

    public VectorMemoryReaderF16(VectorLayout layout, int rowCount) {
        super(layout, rowCount);

    }

    @Override
    public float[] fetchRow(int row) {
        float[] result = new float[dim()];
        widenF16Row(row, result, 0);
        return result;
    }

    private void widenF16Row(int rowId, float[] dst, int off) {
        final int d = dim();
        if (off < 0 || off + d > dst.length)
            throw new IllegalArgumentException("dst slice out of bounds");
        var row = rowSlice(rowId);
        for (int i=0; i < dim(); i++) {
            short h = row.get(H16_LE, (long)i*Short.BYTES);
            dst[i] = Float.float16ToFloat(h);
        }
    }
}
