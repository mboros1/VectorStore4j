package io.github.mboros1.vs4j.core.vectors.memory.reader;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import io.github.mboros1.vs4j.core.vectors.enums.Dtype;
import io.github.mboros1.vs4j.core.vectors.memory.layout.VectorLayout;
import io.github.mboros1.vs4j.core.vectors.memory.writer.RowCursor;
import io.github.mboros1.vs4j.core.vectors.memory.writer.VectorMemoryWriter;
import io.github.mboros1.vs4j.core.vectors.utilities.VectorMath;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.*;

class VectorMemoryReaderTest {

    private static final JsonFactory JSON_FACTORY = new JsonFactory();
    private static final float HALF_TOLERANCE = 5e-4f;

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void readsBackWrittenRows(Dtype dtype, @TempDir Path tempDir) throws Exception {
        int dim = 128; // 128 keeps row bytes aligned for both dtypes
        long rowBytes = (long) dim * dtype.bytes();
        long shardSizeBytes = rowBytes * 2; // 2 rows per shard to exercise rollover

        float[][] source = new float[][] {
                buildDeterministicRow(0, dim),
                buildDeterministicRow(1, dim),
                buildDeterministicRow(2, dim)
        };

        try (VectorMemoryWriter writer = VectorMemoryWriter.create(tempDir, dim, dtype, shardSizeBytes)) {
            for (int i = 0; i < source.length; i++) {
                RowCursor row = writer.allocRow(i);
                row.putArray(source[i], 0);
            }
        }

        VectorLayout layout = VectorLayout.of(tempDir, dim, dtype, shardSizeBytes);
        assertEquals(layout.rowBytes(), layout.rowStride(), "expected no padding when enforcing 64-byte alignment");
        assertEquals(0, layout.rowBytes() % 64, "row bytes must remain 64-byte aligned");

        try (VectorMemoryReader reader = new VectorMemoryReader(layout, source.length)) {
            assertEquals(dim, reader.dim());
            assertEquals(dtype, reader.dtype());
            assertEquals(source.length, reader.numDocs());

            for (int rowIdx = 0; rowIdx < source.length; rowIdx++) {
                MemorySegment slice = reader.rowSlice(rowIdx);
                assertEquals(reader.rowStride(), slice.byteSize(), "unexpected slice length");
                assertEquals(0, slice.byteSize() % 64, "slice size should maintain 64-byte alignment");

                float[] actual = readRow(slice, dtype, dim);
                assertArrayEquals(expectedStoredRow(source[rowIdx], dtype), actual, deltaFor(dtype));
            }

            assertThrows(IndexOutOfBoundsException.class, () -> reader.rowSlice(-1));
            assertThrows(IndexOutOfBoundsException.class, () -> reader.rowSlice(source.length));
        }
    }

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void readsEmbeddedSamplesWithAlignment(Dtype dtype, @TempDir Path tempDir) throws Exception {
        Path embeddedDir = Paths.get("..", "samples", "embedded").toAbsolutePath().normalize();
        assertTrue(Files.isDirectory(embeddedDir), "embedded samples directory missing");

        Path sampleFile;
        try (var stream = Files.list(embeddedDir)) {
            sampleFile = stream.filter(Files::isRegularFile)
                    .sorted()
                    .findFirst()
                    .orElseThrow(() -> new IllegalStateException("no embedded sample files found"));
        }

        EmbeddedSample sample = readEmbeddedSample(sampleFile, 32);
        assertFalse(sample.embeddings().isEmpty(), "no embeddings parsed from sample");

        int dim = sample.dim();
        long shardSizeBytes = (long) dtype.bytes() * dim * 8; // small shards to exercise mapper rollover

        AtomicInteger counter = new AtomicInteger();
        try (VectorMemoryWriter writer = VectorMemoryWriter.create(tempDir, dim, dtype, shardSizeBytes)) {
            for (float[] embedding : sample.embeddings()) {
                RowCursor row = writer.allocRow(counter.getAndIncrement());
                row.putArray(embedding, 0);
            }
        }

        VectorLayout layout = VectorLayout.of(tempDir, dim, dtype, shardSizeBytes);
        assertEquals(layout.rowBytes(), layout.rowStride(), "layout should retain 64-byte row alignment");
        assertEquals(0, layout.rowBytes() % 64, "row bytes expected to be 64-byte aligned");

        try (VectorMemoryReader reader = new VectorMemoryReader(layout, counter.get())) {
            assertEquals(dim, reader.dim());
            assertEquals(dtype, reader.dtype());
            assertEquals(counter.get(), reader.numDocs());

            for (int rowIdx = 0; rowIdx < reader.numDocs(); rowIdx++) {
                MemorySegment slice = reader.rowSlice(rowIdx);
                assertEquals(reader.rowStride(), slice.byteSize());
                assertEquals(0, slice.byteSize() % 64, "row slice should preserve 64-byte alignment");

                float[] actual = readRow(slice, dtype, dim);
                float[] expected = expectedStoredRow(sample.embeddings().get(rowIdx), dtype);
                assertArrayEquals(expected, actual, deltaFor(dtype));
            }
        }
    }

    private float[] readRow(MemorySegment slice, Dtype dtype, int dim) {
        float[] values = new float[dim];
        ValueLayout.OfFloat f32Layout = ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
        for (int i = 0; i < dim; i++) {
            values[i] = slice.get(f32Layout, (long) i * Float.BYTES);
        }
        return values;
    }

    private float[] expectedStoredRow(float[] source, Dtype dtype) {
        float[] normalized = Arrays.copyOf(source, source.length);
        VectorMath.normalize(normalized);
        if (dtype == Dtype.F16) {
            return quantize(normalized);
        }
        return normalized;
    }

    private float[] quantize(float[] row) {
        float[] result = Arrays.copyOf(row, row.length);
        for (int i = 0; i < result.length; i++) {
            result[i] = Float.float16ToFloat(Float.floatToFloat16(result[i]));
        }
        return result;
    }

    private float deltaFor(Dtype dtype) {
        return dtype == Dtype.F16 ? HALF_TOLERANCE : 1e-6f;
    }

    private float[] buildDeterministicRow(int rowIndex, int dim) {
        float[] row = new float[dim];
        for (int i = 0; i < dim; i++) {
            float base = (rowIndex + 1) * 0.5f + (i + 1) * 0.01f;
            row[i] = base;
        }
        return row;
    }

    private EmbeddedSample readEmbeddedSample(Path file, int limit) throws IOException {
        int dim = -1;
        List<float[]> embeddings = new ArrayList<>();

        try (JsonParser parser = JSON_FACTORY.createParser(file.toFile())) {
            if (parser.nextToken() != JsonToken.START_ARRAY) {
                throw new IOException("expected START_ARRAY at root: " + file);
            }

            while (parser.nextToken() != JsonToken.END_ARRAY && embeddings.size() < limit) {
                if (parser.currentToken() != JsonToken.START_OBJECT) {
                    parser.skipChildren();
                    continue;
                }

                float[] embedding = null;
                while (parser.nextToken() != JsonToken.END_OBJECT) {
                    String fieldName = parser.getCurrentName();
                    parser.nextToken();
                    if ("metadata".equals(fieldName) && parser.currentToken() == JsonToken.START_OBJECT) {
                        while (parser.nextToken() != JsonToken.END_OBJECT) {
                            String metaName = parser.getCurrentName();
                            parser.nextToken();
                            if ("embedding".equals(metaName)) {
                                embedding = readEmbedding(parser, dim);
                                if (embedding != null && dim < 0) {
                                    dim = embedding.length;
                                }
                            } else {
                                parser.skipChildren();
                            }
                        }
                    } else {
                        parser.skipChildren();
                    }
                }

                if (embedding != null) {
                    if (dim >= 0 && embedding.length != dim) {
                        throw new IOException("inconsistent embedding length in " + file);
                    }
                    embeddings.add(embedding);
                }
            }
        }

        if (dim <= 0) {
            throw new IOException("no embedding arrays discovered in " + file);
        }

        return new EmbeddedSample(dim, embeddings);
    }

    private float[] readEmbedding(JsonParser parser, int expectedDim) throws IOException {
        if (parser.currentToken() != JsonToken.START_ARRAY) {
            throw new IOException("expected embedding array start");
        }

        if (expectedDim < 0) {
            List<Float> values = new ArrayList<>();
            while (parser.nextToken() != JsonToken.END_ARRAY) {
                float value = parser.currentToken().isNumeric()
                        ? parser.getFloatValue()
                        : Float.parseFloat(parser.getValueAsString());
                values.add(value);
            }
            float[] result = new float[values.size()];
            for (int i = 0; i < values.size(); i++) {
                result[i] = values.get(i);
            }
            return result;
        } else {
            float[] result = new float[expectedDim];
            int idx = 0;
            while (parser.nextToken() != JsonToken.END_ARRAY) {
                if (idx >= expectedDim) {
                    throw new IOException("embedding length exceeds expected " + expectedDim);
                }
                float value = parser.currentToken().isNumeric()
                        ? parser.getFloatValue()
                        : Float.parseFloat(parser.getValueAsString());
                result[idx++] = value;
            }
            if (idx != expectedDim) {
                throw new IOException("embedding length " + idx + " does not match expected " + expectedDim);
            }
            return result;
        }
    }

    private record EmbeddedSample(int dim, List<float[]> embeddings) {}
}
