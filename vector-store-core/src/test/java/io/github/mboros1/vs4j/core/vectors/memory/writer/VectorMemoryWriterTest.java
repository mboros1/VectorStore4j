package io.github.mboros1.vs4j.core.vectors.memory.writer;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import io.github.mboros1.vs4j.core.vectors.enums.Dtype;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.List;
import java.util.Locale;
import java.util.SplittableRandom;
import java.util.concurrent.Callable;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.ExecutorCompletionService;
import java.util.concurrent.Future;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.*;

public class VectorMemoryWriterTest {

    private static final JsonFactory JSON_FACTORY = new JsonFactory();
    private static final float HALF_TOLERANCE = 5e-4f;
    private static final float HALF_MAX_ERROR = 1e-3f;
    private static final int PROPERTY_SAMPLES = 10_000;

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void allocRowAndPutArrayWritesSequentialRows(Dtype dtype, @TempDir Path tempDir) throws Exception {
        AtomicInteger counter = new AtomicInteger();
        int dim = 64;
        float[] first = new float[64];
        float f = 2.0f;
        for(int i = 0; i < 64; i++) {
            first[i] = f;
            f += 1.0f;
        }
        float[] second = new float[64];
        f = 2.0f;
        for(int i = 0; i < 64; i++) {
            second[i] = f;
            f *= i;
        }

        try (VectorMemoryWriter memory = VectorMemoryWriter.create(tempDir, dim, dtype)) {
            RowCursor row0 = memory.allocRow(counter.getAndIncrement());
            RowCursor row1 = memory.allocRow(counter.getAndIncrement());

            assertEquals(0, row0.rowIndex());
            assertEquals(1, row1.rowIndex());
            assertEquals(dim, memory.dim());
            assertEquals(dtype, memory.dtype());

            row0.putArray(first, 0);
            row1.putArray(second, 0);
        }

        float[][] rows = readRows(tempDir, dtype, dim, 2, BaseVectorMemoryWriter.DEFAULT_SHARD_SIZE_BYTES);
        assertArrayEquals(first, rows[0], deltaFor(dtype));
        assertArrayEquals(second, rows[1], deltaFor(dtype));
    }

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void putArrayHonorsSourceOffset(Dtype dtype, @TempDir Path tempDir) throws Exception {
        int dim = 64;
        float[] source = new float[70];
        float f = 1.0f;
        source[0] = 9.0f;
        for (int i = 1; i < 70; i++) {
            source[i] = f;
            f += 1.0f;
        }
        float[] comp = new float[64];
        f = 2.0f;
        for(int i = 0; i < 64; i++) {
            comp[i] = f;
            f += 1.0f;
        }

        try (VectorMemoryWriter memory = VectorMemoryWriter.create(tempDir, dim, dtype)) {
            RowCursor row = memory.allocRow(0);
            row.putArray(source, 2);
        }

        float[][] rows = readRows(tempDir, dtype, dim, 1, BaseVectorMemoryWriter.DEFAULT_SHARD_SIZE_BYTES);
        assertArrayEquals(comp, rows[0], deltaFor(dtype));
    }

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void putBufferWritesDataAndAdvancesSource(Dtype dtype, @TempDir Path tempDir) throws Exception {
        int dim = 64;
        float[] floatArr = new float[70];
        float f = 1.0f;
        for (int i = 0; i < 70; i++) {
            floatArr[i] = f;
            f += 1.0f;
        }
        float[] floatArrComp = new float[64];
        f = 2.0f;
        for(int i = 0; i < 64; i++) {
            floatArrComp[i] = f;
            f += 1.0f;
        }
        FloatBuffer buffer = FloatBuffer.wrap(floatArr);
        buffer.position(1);

        try (VectorMemoryWriter memory = VectorMemoryWriter.create(tempDir, dim, dtype)) {
            RowCursor row = memory.allocRow(0);
            row.putBuffer(buffer);
            assertEquals(1 + dim, buffer.position());
        }

        float[][] rows = readRows(tempDir, dtype, dim, 1, BaseVectorMemoryWriter.DEFAULT_SHARD_SIZE_BYTES);
        assertArrayEquals(floatArrComp, rows[0], deltaFor(dtype));
    }

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void putBufferRejectsInsufficientData(Dtype dtype, @TempDir Path tempDir) throws Exception {
        int dim = 64;
        FloatBuffer buffer = FloatBuffer.wrap(new float[]{1f, 2f, 3f});

        try (VectorMemoryWriter memory = VectorMemoryWriter.create(tempDir, dim, dtype)) {
            RowCursor row = memory.allocRow(0);
            IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, () -> row.putBuffer(buffer));
            assertTrue(ex.getMessage() == null || ex.getMessage().toLowerCase(Locale.ROOT).contains("enough"));
        }
    }

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void putFromJsonArrayParsesMixedNumericForms(Dtype dtype, @TempDir Path tempDir) throws Exception {
        int dim = 64;
        Object[] floatArr = new Object[dim];

        floatArr[0] = 1;
        floatArr[1] = "2e-1";
        floatArr[2] = "3e+1";
        floatArr[3] = -4.5;
        for (int i = 4; i < dim; i++) {
            floatArr[i] = 1.0f * i;
        }
        String floatString = Arrays.asList(floatArr).toString();

        try (VectorMemoryWriter memory = VectorMemoryWriter.create(tempDir, dim, dtype)) {
            RowCursor row = memory.allocRow(0);
            try (JsonParser parser = JSON_FACTORY.createParser(floatString)) {
                parser.nextToken();
                row.putFromJsonArray(parser);
            }
        }

        float[][] rows = readRows(tempDir, dtype, dim, 1, BaseVectorMemoryWriter.DEFAULT_SHARD_SIZE_BYTES);
        var firstFour = Arrays.asList(rows[0]).stream().limit(4).toList();
        float[] firstFourFloats = new float[4];
        for (int i = 0; i < 4; ++i) {
            firstFourFloats[i] = firstFour.get(0)[i];
        }
        assertArrayEquals(new float[]{1f, 0.2f, 30f, -4.5f}, firstFourFloats, deltaFor(dtype));
    }

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void putFromJsonArrayRejectsNestedStructures(Dtype dtype, @TempDir Path tempDir) throws Exception {
        int dim = 64;

        try (VectorMemoryWriter memory = VectorMemoryWriter.create(tempDir, dim, dtype)) {
            RowCursor row = memory.allocRow(0);
            try (JsonParser parser = JSON_FACTORY.createParser("[1, {\"x\":2}, 3]")) {
                parser.nextToken();
                assertThrows(IOException.class, () -> row.putFromJsonArray(parser));
            }
        }
    }

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void putFromJsonArrayRequiresExactLength(Dtype dtype, @TempDir Path tempDir) throws Exception {
        int dim = 64;

        try (VectorMemoryWriter memory = VectorMemoryWriter.create(tempDir, dim, dtype)) {
            RowCursor row = memory.allocRow(0);
            var vectorStringTooShort = IntStream.range(1, 63).boxed().toList().toString();
            var vectorStringTooLong = IntStream.range(1, 68).boxed().toList().toString();

            try (JsonParser shortParser = JSON_FACTORY.createParser(vectorStringTooShort)) {
                shortParser.nextToken();
                IOException ex = assertThrows(IOException.class, () -> row.putFromJsonArray(shortParser));
                assertTrue(ex.getMessage() == null || ex.getMessage().contains("incomplete"));
            }

            try (JsonParser longParser = JSON_FACTORY.createParser(vectorStringTooLong)) {
                longParser.nextToken();
                IOException ex = assertThrows(IOException.class, () -> row.putFromJsonArray(longParser));
                assertTrue(ex.getMessage() == null || ex.getMessage().contains("too many"));
            }
        }
    }

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void shardRolloverCreatesExpectedFiles(Dtype dtype, @TempDir Path tempDir) throws Exception {
        int dim = 64;
        long shardSizeBytes = (long) dtype.bytes() * dim * 2; // force 2 rows/shard
        int rowsToWrite = 5;
        float[][] expected = new float[rowsToWrite][dim];
        AtomicInteger rowCounter = new AtomicInteger(0);
        for (int r = 0; r < rowsToWrite; r++) {
            expected[r] = buildDeterministicRow(r, dim);
        }

        int rowBytes = dim * dtype.bytes();
        int rowsPerShard = rowsPerShard(shardSizeBytes, rowBytes);

        try (VectorMemoryWriter memory = VectorMemoryWriter.create(tempDir, dim, dtype, shardSizeBytes)) {
            for (float[] src : expected) {
                RowCursor row = memory.allocRow(rowCounter.getAndIncrement());
                row.putArray(src, 0);
            }
        }

        int totalShards = (rowsToWrite + rowsPerShard - 1) / rowsPerShard;
        for (int shardId = 0; shardId < totalShards; shardId++) {
            Path shardPath = shardPath(tempDir, dtype, shardId);
            assertTrue(Files.exists(shardPath), "missing shard " + shardPath);
            int rowsInShard = Math.min(rowsPerShard, rowsToWrite - shardId * rowsPerShard);
            assertEquals((long) rowsInShard * rowBytes, Files.size(shardPath));
        }

        float[][] stored = readRows(tempDir, dtype, dim, rowsToWrite, shardSizeBytes);
        float delta = deltaFor(dtype);
        for (int i = 0; i < rowsToWrite; i++) {
            float[] expectedRow = dtype == Dtype.F16
                    ? quantizeRow(expected[i])
                    : expected[i];
            assertArrayEquals(expectedRow, stored[i], delta);
        }
    }

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void concurrentAllocationsProduceUniqueRowIds(Dtype dtype, @TempDir Path tempDir) throws Exception {
        int dim = 128;
        int threads = Math.min(8, Runtime.getRuntime().availableProcessors());
        int rowsPerThread = 64;
        long shardSizeBytes = (long) dtype.bytes() * dim * 32;
        int totalRows = threads * rowsPerThread;
        AtomicInteger rowCount = new AtomicInteger(0);

        ThreadPoolExecutor executor = newExecutor(threads);
        try (VectorMemoryWriter memory = VectorMemoryWriter.create(tempDir, dim, dtype, shardSizeBytes)) {
            List<Callable<int[]>> tasks = new ArrayList<>();
            SplittableRandom seeds = new SplittableRandom(0x5eed);
            for (int t = 0; t < threads; t++) {
                long seed = seeds.nextLong();
                tasks.add(() -> writeRandomRows(memory, dim, rowsPerThread, seed, rowCount));
            }

            ExecutorCompletionService<int[]> completion = new ExecutorCompletionService<>(executor);
            tasks.forEach(completion::submit);

            BitSet seen = new BitSet(totalRows);
            int maxIndex = -1;
            for (int i = 0; i < tasks.size(); i++) {
                int[] indexes = completion.take().get();
                for (int idx : indexes) {
                    assertFalse(seen.get(idx), "duplicate index " + idx);
                    seen.set(idx);
                    maxIndex = Math.max(maxIndex, idx);
                }
            }

            assertEquals(totalRows, seen.cardinality());
            assertEquals(totalRows - 1, maxIndex);
        } finally {
            shutdownExecutor(executor);
        }
    }

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void concurrentAllocationsWithInterleavingMaintainOrdering(Dtype dtype, @TempDir Path tempDir) throws Exception {
        int dim = 64;
        int threads = Math.min(6, Runtime.getRuntime().availableProcessors());
        int rowsPerThread = 32;
        long shardSizeBytes = (long) dtype.bytes() * dim * 16;
        int totalRows = threads * rowsPerThread;
        AtomicInteger rowCount = new AtomicInteger(0);

        ThreadPoolExecutor executor = newExecutor(threads);
        try (VectorMemoryWriter memory = VectorMemoryWriter.create(tempDir, dim, dtype, shardSizeBytes)) {
            List<Callable<int[]>> tasks = new ArrayList<>();
            SplittableRandom seeds = new SplittableRandom(0x1ced);
            CyclicBarrier barrier = new CyclicBarrier(threads);
            for (int t = 0; t < threads; t++) {
                long seed = seeds.nextLong();
                tasks.add(() -> writeRowsWithInterleaving(memory, dim, rowsPerThread, seed, barrier, rowCount));
            }

            ExecutorCompletionService<int[]> completion = new ExecutorCompletionService<>(executor);
            tasks.forEach(completion::submit);

            BitSet seen = new BitSet(totalRows);
            for (int i = 0; i < tasks.size(); i++) {
                int[] indexes = completion.take().get();
                for (int idx : indexes) {
                    assertFalse(seen.get(idx), "duplicate index " + idx);
                    seen.set(idx);
                }
            }

            assertEquals(totalRows, seen.cardinality());
        } finally {
            shutdownExecutor(executor);
        }
    }

    @Test
    void halfPrecisionQuantizationBoundedError(@TempDir Path tempDir) throws Exception {
        int dim = 32;
        int rows = (PROPERTY_SAMPLES + dim - 1) / dim;
        long shardSizeBytes = (long) Dtype.F16.bytes() * dim * 64;
        SplittableRandom rng = new SplittableRandom(0xF16F16);

        float[] source = new float[rows * dim];
        for (int i = 0; i < source.length; i++) {
            source[i] = (float) (rng.nextDouble(-1.0, 1.0));
        }

        try (VectorMemoryWriter memory = VectorMemoryWriter.create(tempDir, dim, Dtype.F16, shardSizeBytes)) {
            for (int row = 0; row < rows; row++) {
                RowCursor cursor = memory.allocRow(row);
                cursor.putArray(source, row * dim);
            }
        }

        float[][] stored = readRows(tempDir, Dtype.F16, dim, rows, shardSizeBytes);
        double maxError = 0.0;
        int overTolerance = 0;
        int idx = 0;
        for (float[] storedRow : stored) {
            for (float value : storedRow) {
                float original = source[idx++];
                double error = Math.abs(original - value);
                maxError = Math.max(maxError, error);
                if (Math.abs(original) >= 1e-3) {
                    assertTrue(error <= HALF_TOLERANCE, "error " + error + " too large for " + original);
                } else if (error > HALF_TOLERANCE) {
                    overTolerance++;
                }
            }
        }
        assertTrue(maxError <= HALF_MAX_ERROR, "max error " + maxError + " exceeds bound");
        assertTrue(overTolerance <= PROPERTY_SAMPLES * 0.01, "unexpected high error count " + overTolerance);
    }

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void embeddedSamplesLoadWithExecutor(Dtype dtype, @TempDir Path tempDir) throws Exception {
        Path embeddedDir = Paths.get("..", "samples", "embedded").toAbsolutePath().normalize();
        assertTrue(Files.isDirectory(embeddedDir), "embedded samples directory missing");

        List<Path> files = listFiles(embeddedDir);
        assertFalse(files.isEmpty(), "no embedded sample files found");

        int dim = determineEmbeddingDim(files.getFirst());
        long shardSizeBytes = (long) dtype.bytes() * dim * 64; // force multiple shards for corpus
        int sampleWindow = 16;

        ConcurrentSampleWindow sampleWindowRows = new ConcurrentSampleWindow(sampleWindow, dim);
        AtomicInteger totalRows = new AtomicInteger();

        ThreadPoolExecutor executor = newExecutor(Math.min(files.size(), Runtime.getRuntime().availableProcessors()));
        try (VectorMemoryWriter memory = VectorMemoryWriter.create(tempDir, dim, dtype, shardSizeBytes)) {
            ExecutorCompletionService<Void> completion = new ExecutorCompletionService<>(executor);
            for (Path file : files) {
                completion.submit(() -> {
                    ingestEmbeddedFile(file, memory, dim, sampleWindowRows, totalRows);
                    return null;
                });
            }

            for (int i = 0; i < files.size(); i++) {
                Future<Void> future = completion.take();
                future.get();
            }
        } finally {
            shutdownExecutor(executor);
        }

        int sampleCount = sampleWindowRows.size();
        assertTrue(totalRows.get() >= sampleCount && sampleCount > 0);

        float[][] stored = readRows(tempDir, dtype, dim, sampleCount, shardSizeBytes);
        float delta = deltaFor(dtype);
        for (int i = 0; i < sampleCount; i++) {
            float[] expectedRow = dtype == Dtype.F16
                    ? quantizeRow(sampleWindowRows.row(i))
                    : sampleWindowRows.row(i);
            assertArrayEquals(expectedRow, stored[i], delta);
        }
    }

    private float deltaFor(Dtype dtype) {
        return dtype == Dtype.F16 ? HALF_TOLERANCE : 1e-6f;
    }

    private float[][] readRows(Path bundlePath, Dtype dtype, int dim, int rows, long shardSizeBytes) throws IOException {
        int rowBytes = dim * dtype.bytes();
        int rowsPerShard = rowsPerShard(shardSizeBytes, rowBytes);
        int shardCount = (rows + rowsPerShard - 1) / rowsPerShard;

        float[][] result = new float[rows][dim];
        int rowIndex = 0;
        for (int shardId = 0; shardId < shardCount; shardId++) {
            Path shardPath = shardPath(bundlePath, dtype, shardId);
            assertTrue(Files.exists(shardPath), STR."missing shard \{shardPath}");
            try (FileChannel fc = FileChannel.open(shardPath, StandardOpenOption.READ)) {
                for (int i = 0; i < rowsPerShard && rowIndex < rows; i++) {
                    ByteBuffer buffer = ByteBuffer.allocate(rowBytes);
                    readFully(fc, buffer, (long) i * rowBytes);
                    buffer.order(ByteOrder.LITTLE_ENDIAN);
                    float[] dest = result[rowIndex++];
                    buffer.rewind();
                    if (dtype == Dtype.F32) {
                        for (int c = 0; c < dim; c++) {
                            dest[c] = buffer.getFloat();
                        }
                    } else {
                        for (int c = 0; c < dim; c++) {
                            dest[c] = Float.float16ToFloat(buffer.getShort());
                        }
                    }
                }
            }
        }
        return result;
    }

    private int determineEmbeddingDim(Path file) throws IOException {
        try (JsonParser parser = JSON_FACTORY.createParser(file.toFile())) {
            if (parser.nextToken() != JsonToken.START_ARRAY) {
                throw new IOException("expected START_ARRAY at root: " + file);
            }
            while (parser.nextToken() != JsonToken.END_ARRAY) {
                if (parser.currentToken() != JsonToken.START_OBJECT) {
                    parser.skipChildren();
                    continue;
                }
                while (parser.nextToken() != JsonToken.END_OBJECT) {
                    String fieldName = parser.getCurrentName();
                    parser.nextToken();
                    if ("metadata".equals(fieldName) && parser.currentToken() == JsonToken.START_OBJECT) {
                        while (parser.nextToken() != JsonToken.END_OBJECT) {
                            String metaField = parser.getCurrentName();
                            parser.nextToken();
                            if ("embedding".equals(metaField)) {
                                return readEmbeddingLength(parser);
                            } else {
                                parser.skipChildren();
                            }
                        }
                    } else {
                        parser.skipChildren();
                    }
                }
            }
        }
        throw new IOException("no embedding array found in " + file);
    }

    private void ingestEmbeddedFile(Path path,
                                    VectorMemoryWriter memory,
                                    int dim,
                                    ConcurrentSampleWindow sampleWindow,
                                    AtomicInteger totalRows) throws IOException {
        ThreadLocal<float[]> buffer = ThreadLocal.withInitial(() -> new float[dim]);
        try (JsonParser parser = JSON_FACTORY.createParser(path.toFile())) {
            if (parser.nextToken() != JsonToken.START_ARRAY) {
                throw new IOException("expected START_ARRAY at root: " + path);
            }

            while (parser.nextToken() != JsonToken.END_ARRAY) {
                if (parser.currentToken() != JsonToken.START_OBJECT) {
                    parser.skipChildren();
                    continue;
                }

                while (parser.nextToken() != JsonToken.END_OBJECT) {
                    String fieldName = parser.getCurrentName();
                    parser.nextToken();
                    if ("metadata".equals(fieldName) && parser.currentToken() == JsonToken.START_OBJECT) {
                        while (parser.nextToken() != JsonToken.END_OBJECT) {
                            String metaField = parser.getCurrentName();
                            parser.nextToken();
                            if ("embedding".equals(metaField)) {
                                float[] local = buffer.get();
                                readEmbedding(parser, dim, local);
                                var currRow = totalRows.getAndIncrement();
                                RowCursor row = memory.allocRow(currRow);
                                row.putArray(local, 0);
                                sampleWindow.capture(row.rowIndex(), local);
                            } else {
                                parser.skipChildren();
                            }
                        }
                    } else {
                        parser.skipChildren();
                    }
                }
            }
        } catch (IOException e) {
            throw new IOException("Failed processing " + path + ": " + e.getMessage(), e);
        }
    }

    private void readEmbedding(JsonParser parser, int dim, float[] buffer) throws IOException {
        if (parser.currentToken() != JsonToken.START_ARRAY) {
            throw new IOException("expected START_ARRAY for embedding");
        }
        int i = 0;
        while (parser.nextToken() != JsonToken.END_ARRAY) {
            float value = parser.currentToken().isNumeric()
                    ? parser.getFloatValue()
                    : Float.parseFloat(parser.getValueAsString());
            if (i >= dim) {
                throw new IOException("embedding length exceeds expected " + dim);
            }
            buffer[i++] = value;
        }
        if (i != dim) {
            throw new IOException("embedding length " + i + " does not match expected " + dim);
        }
    }

    private int readEmbeddingLength(JsonParser parser) throws IOException {
        if (parser.currentToken() != JsonToken.START_ARRAY) {
            throw new IOException("expected START_ARRAY for embedding");
        }
        int count = 0;
        while (parser.nextToken() != JsonToken.END_ARRAY) {
            parser.skipChildren();
            count++;
        }
        return count;
    }

    private static int rowsPerShard(long shardSizeBytes, int rowBytes) {
        return (int) Math.max(1L, shardSizeBytes / rowBytes);
    }

    private static Path shardPath(Path bundlePath, Dtype dtype, int shardId) {
        return bundlePath.resolve("vectors" + shardId + "." + dtype.name().toLowerCase(Locale.ROOT));
    }

    private static void readFully(FileChannel fc, ByteBuffer buffer, long position) throws IOException {
        buffer.clear();
        int read;
        int offset = 0;
        while (buffer.hasRemaining()) {
            read = fc.read(buffer, position + offset);
            if (read < 0) {
                throw new EOFException("unexpected EOF while reading row");
            }
            offset += read;
        }
        buffer.flip();
    }

    private static float[] buildDeterministicRow(int rowIndex, int dim) {
        float[] row = new float[dim];
        for (int i = 0; i < dim; i++) {
            row[i] = rowIndex + (i + 1) * 0.1f;
        }
        return row;
    }

    private static float[] quantizeRow(float[] row) {
        float[] result = Arrays.copyOf(row, row.length);
        for (int i = 0; i < result.length; i++) {
            result[i] = Float.float16ToFloat(Float.floatToFloat16(result[i]));
        }
        return result;
    }

    private static int[] writeRandomRows(VectorMemoryWriter memory, int dim, int count, long seed, AtomicInteger rowCounter) {
        SplittableRandom rng = new SplittableRandom(seed);
        float[] buffer = new float[dim];
        int[] indexes = new int[count];
        for (int i = 0; i < count; i++) {
            fillRandom(buffer, rng);
            RowCursor row = memory.allocRow(rowCounter.getAndIncrement());
            row.putArray(buffer, 0);
            indexes[i] = row.rowIndex();
        }
        return indexes;
    }

    private static int[] writeRowsWithInterleaving(VectorMemoryWriter memory,
                                                   int dim,
                                                   int count,
                                                   long seed,
                                                   CyclicBarrier barrier, AtomicInteger rowCounter) throws Exception {
        SplittableRandom rng = new SplittableRandom(seed);
        float[] buffer = new float[dim];
        int[] indexes = new int[count];
        for (int i = 0; i < count; i++) {
            fillRandom(buffer, rng);
            if (rng.nextBoolean()) {
                RowCursor row = memory.allocRow(rowCounter.getAndIncrement());
                barrier.await();
                row.putArray(buffer, 0);
                indexes[i] = row.rowIndex();
            } else {
                barrier.await();
                RowCursor row = memory.allocRow(rowCounter.getAndIncrement());
                row.putArray(buffer, 0);
                indexes[i] = row.rowIndex();
            }
        }
        return indexes;
    }

    private static void fillRandom(float[] buffer, SplittableRandom rng) {
        for (int i = 0; i < buffer.length; i++) {
            buffer[i] = (float) rng.nextDouble(-1.0, 1.0);
        }
    }

    private static ThreadPoolExecutor newExecutor(int workers) {
        ThreadPoolExecutor executor = new ThreadPoolExecutor(
                workers,
                workers,
                0L,
                TimeUnit.MILLISECONDS,
                new LinkedBlockingQueue<>(),
                r -> {
                    Thread t = new Thread(r);
                    t.setDaemon(true);
                    t.setName("vector-memory-test-worker");
                    return t;
                });
        executor.prestartAllCoreThreads();
        return executor;
    }

    private static void shutdownExecutor(ThreadPoolExecutor executor) {
        executor.shutdown();
        try {
            if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                executor.shutdownNow();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            executor.shutdownNow();
        }
    }

    private static List<Path> listFiles(Path dir) throws IOException {
        try (var stream = Files.list(dir)) {
            return stream.filter(Files::isRegularFile)
                    .sorted()
                    .toList();
        }
    }

    private static final class ConcurrentSampleWindow {
        private final float[][] samples;
        private final BitSet captured = new BitSet();
        private final int dim;
        private volatile int size;

        ConcurrentSampleWindow(int maxSamples, int dim) {
            this.dim = dim;
            this.samples = new float[maxSamples][];
            for (int i = 0; i < maxSamples; i++) {
                samples[i] = new float[dim];
            }
        }

        void capture(int rowIndex, float[] values) {
            if (rowIndex >= samples.length) {
                return;
            }
            synchronized (captured) {
                if (captured.get(rowIndex)) {
                    return;
                }
                captured.set(rowIndex);
                System.arraycopy(values, 0, samples[rowIndex], 0, dim);
                size = Math.max(size, rowIndex + 1);
            }
        }

        int size() {
            return size;
        }

        float[] row(int index) {
            synchronized (captured) {
                if (!captured.get(index)) {
                    throw new IllegalStateException("row " + index + " not captured");
                }
                return Arrays.copyOf(samples[index], dim);
            }
        }
    }
}
