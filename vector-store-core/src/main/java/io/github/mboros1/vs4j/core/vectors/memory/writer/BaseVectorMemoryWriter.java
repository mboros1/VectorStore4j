package io.github.mboros1.vs4j.core.vectors.memory.writer;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;
import io.github.mboros1.vs4j.core.vectors.memory.ShardMapper;
import io.github.mboros1.vs4j.core.vectors.memory.layout.VectorLayout;

import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;

sealed abstract class BaseVectorMemoryWriter implements VectorMemoryWriter permits VectorMemoryWriterF32, VectorMemoryWriterF16 {
    static final long DEFAULT_SHARD_SIZE_BYTES = 1L << 30; // 1 GiB
    private final VectorLayout layout;
    private final AtomicInteger rowCounter = new AtomicInteger(0);
    private final ShardMapper shardMapper;

    public BaseVectorMemoryWriter(VectorLayout layout) {
        if (layout.rowBytes() != layout.rowStride())
            throw new IllegalArgumentException(STR."Row stride must be aligned without padding, Row Stride: \{layout.rowsPerShard()}, Row Bytes: \{layout.rowBytes()}");
        this.layout = layout;
        this.shardMapper = new ShardMapper(layout, ShardMapper.Mode.READ_WRITE);
    }

    protected abstract RowCursor newRow(int rowId, MemorySegment rowSeg, int rowDim);

    @Override
    public final RowCursor allocRow(int rowId) {
        rowCounter.incrementAndGet();
        var rowSeg = shardMapper.rowSlice(rowId);
        return newRow(rowId, rowSeg, dim());
    }

    @Override
    public int dim() {
        return layout.dim();
    }

    @Override
    public void close() throws Exception {
        try {
            shardMapper.close();
        } finally {
            shardMapper.trimLastShard(rowCounter.get());
        }
    }

    @Override
    public Dtype dtype() {
        return layout.dtype();
    }
}
