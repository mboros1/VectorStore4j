package io.github.mboros1.vs4j.core.vectors.memory;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

abstract class BaseVectorMemory implements VectorMemory {
    protected static final int shardSize = 1024 * 1024 * 1024;
    protected final int rowsPerShard;
    protected final int rowBytes;
    protected final Dtype dtype;
    protected final int rowDim;
    protected final AtomicInteger currentRow = new AtomicInteger(0);
    protected final ConcurrentHashMap<Integer, MemorySegment> shards = new ConcurrentHashMap<>();
    protected final Path bundlePath;
    private final Arena arena = Arena.ofShared();

    public BaseVectorMemory(Path bundlePath, int dim, Dtype dtype) {
        this.bundlePath = bundlePath;
        this.rowDim = dim;
        this.dtype = dtype;
        this.rowBytes = rowDim * dtype.bytes();
        this.rowsPerShard = shardSize / rowBytes;
    }

    protected abstract RowCursor newRow(int rowId, MemorySegment rowSeg, int rowDim);

    public final RowCursor allocRow() {
        int rowId = currentRow.getAndIncrement();
        int shardId = rowId / rowsPerShard;
        int idxInShard = rowId % rowsPerShard;
        MemorySegment shard = shardFor(shardId);
        long offset = (long) idxInShard * rowBytes;
        MemorySegment rowSeg = shard.asSlice(offset, rowBytes);
        return newRow(rowId, rowSeg, rowDim);  // monomorphic → inlined
    }

    private Path vectorsPathFor(int shardId) {
        return bundlePath.resolve("vectors" + shardId + "." + dtype.name().toLowerCase());
    }

    private MemorySegment shardFor(int shardId) {
        return shards.computeIfAbsent(shardId, k -> mapShard(shardId));
    }

    private MemorySegment mapShard(int shardId) {
        Path p = vectorsPathFor(shardId);

        try (FileChannel fc = FileChannel.open(p,
                StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            fc.truncate(shardSize);
            return fc.map(FileChannel.MapMode.READ_WRITE, 0, shardSize, arena);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public int dim() {
        return rowDim;
    }

    public void close() throws Exception {

    }
}
