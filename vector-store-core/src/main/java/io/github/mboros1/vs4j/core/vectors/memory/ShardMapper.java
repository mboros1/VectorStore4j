package io.github.mboros1.vs4j.core.vectors.memory;

import io.github.mboros1.vs4j.core.vectors.memory.layout.VectorLayout;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Locale;
import java.util.concurrent.ConcurrentHashMap;

public class ShardMapper implements AutoCloseable {


    private boolean arenaClosed = false;

    public enum Mode { READ_WRITE, READ_ONLY;}
    private final VectorLayout layout;

    private final Mode mode;
    private final Arena arena = Arena.ofShared();
    protected final ConcurrentHashMap<Integer, MemorySegment> shards = new ConcurrentHashMap<>();
    public ShardMapper(VectorLayout layout, Mode mode) {
        this.layout = layout;
        this.mode = mode;
    }

    public Path shardPath(int shardId) {
        return layout.bundlePath().resolve(STR."vectors\{shardId}.\{layout.dtype().name().toLowerCase(Locale.ROOT)}");
    }

    public MemorySegment shard(int shardId) {
        return shards.computeIfAbsent(shardId, this::mapShard);
    }

    private MemorySegment mapShard(int shardId) {
        return switch (mode) {
            case READ_WRITE -> mapShardWrite(shardId);
            case READ_ONLY -> mapShardRead(shardId);
        };
    }

    private MemorySegment mapShardRead(int shardId) {
        Path p = shardPath(shardId);

        try (FileChannel fc = FileChannel.open(p, StandardOpenOption.READ)) {
            return fc.map(FileChannel.MapMode.READ_ONLY, 0, fc.size(), arena);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    private MemorySegment mapShardWrite(int shardId) {
        Path p = shardPath(shardId);

        try (FileChannel fc = FileChannel.open(p,
                StandardOpenOption.CREATE, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            fc.truncate(layout.shardSizeBytes());
            return fc.map(FileChannel.MapMode.READ_WRITE, 0, layout.shardSizeBytes(), arena);
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public MemorySegment rowSlice(int rowId) {
        int shardId = layout.shardOf(rowId);
        int idx     = layout.indexInShard(rowId);
        long off    = (long) idx * layout.rowBytes();
        return shard(shardId).asSlice(off, layout.rowBytes());
    }


    @Override
    public void close() throws Exception {
        arena.close();
        arenaClosed = true;
    }

    public void trimLastShard(int i) {
        if (!arenaClosed) {
            throw new IllegalStateException("Close mapper before trying to trim last shard");
        }
        int totalRows = Math.max(0, i);
        if (totalRows == 0) return;
        int lastShardId = (totalRows - 1) / layout.rowsPerShard();
        int remainder   = totalRows % layout.rowsPerShard();
        int rowsInLast  = (remainder == 0) ? layout.rowsPerShard() : remainder;
        long targetSize = (long) rowsInLast * layout.rowBytes();

        Path last = shardPath(lastShardId);
        try (FileChannel fc = FileChannel.open(last, StandardOpenOption.READ, StandardOpenOption.WRITE)) {
            long current = fc.size();
            if (current > targetSize) {
                fc.truncate(targetSize);
                fc.force(true);
            } else if (current < targetSize) {
                throw new IllegalStateException(STR."Last shard smaller than expected: \{current} < \{targetSize}");
            }
        } catch (IOException e) { throw new UncheckedIOException(e); }
    }
}
