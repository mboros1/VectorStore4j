package io.github.mboros1.vs4j.core.vectors.memory.reader;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;
import io.github.mboros1.vs4j.core.vectors.memory.ShardMapper;
import io.github.mboros1.vs4j.core.vectors.memory.layout.VectorLayout;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

public class VectorMemoryReader implements AutoCloseable {
    private final VectorLayout layout;
    private final ShardMapper shardMapper;
    private final int rowCount;

    public VectorMemoryReader(VectorLayout layout, int rowCount) {
        this.shardMapper = new ShardMapper(layout, ShardMapper.Mode.READ_ONLY);
        this.layout = layout;
        this.rowCount = rowCount;
        validateShards(layout, rowCount);
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
        return shardMapper.rowSlice(rowIdx);
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
