package io.github.mboros1.vs4j.core.vectors.memory.reader;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;
import io.github.mboros1.vs4j.core.vectors.memory.ShardMapper;
import io.github.mboros1.vs4j.core.vectors.memory.layout.VectorLayout;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.VarHandle;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;

import static io.github.mboros1.vs4j.core.vectors.enums.Dtype.F16;

public class VectorMemoryReader implements AutoCloseable {
    private final VectorLayout layout;
    private final ShardMapper shardMapper;
    private final int rowCount;

    private final Arena f32Arena;
    private final MemorySegment f32Cache;
    private final int rowStrideF32;

    public VectorMemoryReader(VectorLayout layout, int rowCount) {
        this.shardMapper = new ShardMapper(layout, ShardMapper.Mode.READ_ONLY);
        this.layout = layout;
        this.rowCount = rowCount;
        validateShards(layout, rowCount);

        if (layout.dtype() == F16) {
            this.rowStrideF32 = align(layout.dim() * Float.BYTES, 64);
            this.f32Arena = Arena.ofShared(); // separate from ShardMapper
            long totalBytes = (long) rowCount * rowStrideF32;
            this.f32Cache = f32Arena.allocate(totalBytes, 64);
            parallelExpandAllF16ToF32();      // fill f32Cache once
        } else {
            this.rowStrideF32 = align(layout.dim() * Float.BYTES, 64);
            this.f32Arena = null;
            this.f32Cache = null;
        }
    }

    private static int align(int n, int a) {
        return ((n + (a - 1)) / a) * a;
    }

    private void parallelExpandAllF16ToF32() {
        final int dim = layout.dim();
        final int rps = layout.rowsPerShard();
        final int shards = (rowCount == 0) ? 0 : ((rowCount - 1) / rps + 1);

        // simple fixed pool sized to Runtime.getRuntime().availableProcessors()
        var pool = java.util.concurrent.Executors.newWorkStealingPool(
                Math.max(1, Runtime.getRuntime().availableProcessors()));
        var tasks = getRunnables(shards, rps, dim);
        tasks.forEach(pool::execute);
        pool.shutdown();
        try { pool.awaitTermination(Long.MAX_VALUE, java.util.concurrent.TimeUnit.SECONDS); }
        catch (InterruptedException ie) { Thread.currentThread().interrupt(); throw new RuntimeException(ie); }
    }

    private ArrayList<Runnable> getRunnables(int shards, int rps, int dim) {
        var tasks = new ArrayList<Runnable>(shards);

        for (int sid = 0; sid < shards; sid++) {
            final int shardId = sid;
            tasks.add(() -> {
                MemorySegment f16Shard = shardMapper.shard(shardId);
                int first = shardId * rps;
                int last  = Math.min(rowCount, first + rps);
                for (int row = first; row < last; row++) {
                    int idxInShard = layout.indexInShard(row);
                    long srcOff = (long) idxInShard * layout.rowBytes(); // f16 stride
                    long dstOff = (long) row * rowStrideF32;

                    MemorySegment src = f16Shard.asSlice(srcOff, layout.rowBytes());
                    MemorySegment dst = f32Cache.asSlice(dstOff, rowStrideF32);
                    // scalar convert f16->f32 into dst (fill exactly dim floats)
                    scalarExpandIntoF32(src, dst, dim);
                }
            });
        }
        return tasks;
    }

    public static void scalarExpandIntoF32(MemorySegment src, MemorySegment dst, int dim) {
        ValueLayout.OfShort layoutF16 = ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
        ValueLayout.OfFloat layoutF32 = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

        long so = 0;
        long doff = 0;
        for (int i = 0; i < dim; i++, so += 2, doff += 4) {
            short h = src.get(layoutF16, (long) i * Short.BYTES);
            float fb = Float.float16ToFloat(h);
            dst.set(layoutF32, doff, fb);
        }
    }

    public int dim() {
        return layout.dim();
    }

    @Override
    public void close() throws Exception {
        shardMapper.close();
    }

    public Dtype dtype() {
        return layout.dtype();
    }
    public MemorySegment rowSlice(int rowIdx) {
        if (rowIdx >= rowCount || rowIdx < 0) {
            throw new IndexOutOfBoundsException("Row index out of bounds");
        }
        return switch (layout.dtype()) {
            case F16 -> f32Cache.asSlice((long) rowIdx * rowStrideF32, rowStrideF32);
            case F32 -> shardMapper.rowSlice(rowIdx);
        };
    }

    public int numDocs() {
        return rowCount;
    }

    private void validateShards(VectorLayout layout, int rowCount) {
        int rps = layout.rowsPerShard();
        long rb  = layout.rowBytes();
        int expectedShards = (rowCount == 0) ? 0 : ( (rowCount - 1) / rps + 1 );

        for (int sid = 0; sid < expectedShards; sid++) {
            int rowsInShard = sid == expectedShards - 1
                    ? (rowCount - 1) % rps + 1
                    : rps;
            long expectedSize = rowsInShard * rb;

            Path p = shardMapper.shardPath(sid); // expose this on ShardMapper
            if (!Files.exists(p)) throw new IllegalStateException(STR."Missing shard: \{p}");

            try (FileChannel fc = FileChannel.open(p, StandardOpenOption.READ)) {
                long actual = fc.size();
                if (actual != expectedSize) {
                    throw new IllegalStateException(STR."Shard \{sid} size mismatch. expected=\{expectedSize} actual=\{actual} path=\{p}");
                }
            } catch (IOException e) {
                throw new UncheckedIOException(STR."Validating shard \{p}", e);
            }
        }
    }
}
