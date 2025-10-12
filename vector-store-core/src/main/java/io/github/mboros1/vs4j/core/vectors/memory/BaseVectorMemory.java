package io.github.mboros1.vs4j.core.vectors.memory;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

sealed abstract class BaseVectorMemory implements VectorMemory permits VectorMemoryF32, VectorMemoryF16 {
    static final long DEFAULT_SHARD_SIZE_BYTES = 1L << 30; // 1 GiB

    protected final long shardSizeBytes;
    protected final int rowBytes;
    protected final Dtype dtype;
    protected final int rowDim;
    protected final int rowsPerShard;
    protected final AtomicInteger currentRow = new AtomicInteger(0);
    protected final ConcurrentHashMap<Integer, MemorySegment> shards = new ConcurrentHashMap<>();
    protected final Path bundlePath;
    private final Arena arena = Arena.ofShared();

    public BaseVectorMemory(Path bundlePath, int dim, Dtype dtype) {
        this(bundlePath, dim, dtype, DEFAULT_SHARD_SIZE_BYTES);
    }

    public BaseVectorMemory(Path bundlePath, int dim, Dtype dtype, long shardSizeBytes) {
        this.bundlePath = bundlePath;
        try { Files.createDirectories(bundlePath); }
        catch (IOException e) { throw new UncheckedIOException(e); }
        this.rowDim = dim;
        this.dtype = dtype;
        this.rowBytes = rowDim * dtype.bytes();
        if (rowBytes <= 0) throw new IllegalArgumentException("invalid row size: " + rowBytes);

        long normalizedShardSize = Math.max(shardSizeBytes, (long) rowBytes);
        this.shardSizeBytes = normalizedShardSize;
        rowsPerShard = Math.toIntExact(Math.max(1L, normalizedShardSize / rowBytes));
    }

    protected abstract RowCursor newRow(int rowId, MemorySegment rowSeg, int rowDim);

    @Override
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
        return bundlePath.resolve("vectors" + shardId + "." + dtype.name().toLowerCase(Locale.ROOT));
    }

    private MemorySegment shardFor(int shardId) {
        return shards.computeIfAbsent(shardId, k -> mapShard(shardId));
    }

    private MemorySegment mapShard(int shardId) {
        Path p = vectorsPathFor(shardId);

        try (FileChannel fc = FileChannel.open(p,
                StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            fc.truncate(shardSizeBytes);
            return fc.map(FileChannel.MapMode.READ_WRITE, 0, shardSizeBytes, arena);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    @Override
    public int dim() {
        return rowDim;
    }

    @Override
    public void close() throws Exception {
        try {
            arena.close();
        } finally {
            trimShardFiles();
        }
    }

    @Override
    public Dtype dtype() {
        return dtype;
    }

    private void trimShardFiles() {
        int totalRows = currentRow.get();
        if (totalRows <= 0) return;

        int totalShards = Math.toIntExact((totalRows + (long) rowsPerShard - 1) / rowsPerShard);

        for (int shardId = 0; shardId < totalShards; shardId++) {
            int rowsInShard = rowsPerShard;
            if (shardId == totalShards - 1) {
                int remainder = totalRows % rowsPerShard;
                if (remainder != 0) rowsInShard = remainder;
            }
            long targetSize = (long) rowsInShard * rowBytes;
            Path path = vectorsPathFor(shardId);
            if (!Files.exists(path)) continue;
            try (FileChannel fc = FileChannel.open(path, StandardOpenOption.WRITE)) {
                fc.truncate(targetSize);
            } catch (IOException e) {
                throw new UncheckedIOException(e);
            }
        }
    }
}
