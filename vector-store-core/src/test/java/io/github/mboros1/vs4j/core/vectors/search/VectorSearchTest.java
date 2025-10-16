package io.github.mboros1.vs4j.core.vectors.search;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;
import io.github.mboros1.vs4j.core.vectors.memory.layout.VectorLayout;
import io.github.mboros1.vs4j.core.vectors.memory.reader.VectorMemoryReader;
import io.github.mboros1.vs4j.core.vectors.memory.writer.VectorMemoryWriter;
import io.github.mboros1.vs4j.core.vectors.utilities.VectorMath;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.*;

public class VectorSearchTest {

    @Test
    public void constructorRejectsNonPositiveK() {
        assertThrows(IllegalArgumentException.class, () -> new VectorSearch.TopKHeap(0));
        assertThrows(IllegalArgumentException.class, () -> new VectorSearch.TopKHeap(-1));
    }

    @Test
    public void offerMaintainsLargestScores() {
        var heap = new VectorSearch.TopKHeap(3);

        assertTrue(heap.offer(1, 0.5f));
        assertEquals(1, heap.size);
        assertEquals(0.5f, heap.minScore());

        assertTrue(heap.offer(2, 0.3f));
        assertEquals(2, heap.size);
        assertEquals(0.3f, heap.minScore());

        assertTrue(heap.offer(3, 0.7f));
        assertEquals(3, heap.size);
        assertEquals(0.3f, heap.minScore());

        assertFalse(heap.offer(4, 0.2f));
        assertEquals(3, heap.size);
        assertEquals(0.3f, heap.minScore());

        assertFalse(heap.offer(5, 0.8f));
        assertEquals(3, heap.size);
        assertEquals(0.5f, heap.minScore());

        assertEquals(1, heap.popMinId());
        assertEquals(2, heap.size);
        assertEquals(0.7f, heap.minScore());

        assertEquals(3, heap.popMinId());
        assertEquals(1, heap.size);
        assertEquals(0.8f, heap.minScore());

        assertEquals(5, heap.popMinId());
        assertEquals(0, heap.size);
    }

    @Test
    public void offerRejectsNonFiniteScores() {
        var heap = new VectorSearch.TopKHeap(2);

        assertFalse(heap.offer(0, Float.NaN));
        assertEquals(0, heap.size);

        assertFalse(heap.offer(1, Float.POSITIVE_INFINITY));
        assertEquals(0, heap.size);

        assertFalse(heap.offer(2, Float.NEGATIVE_INFINITY));
        assertEquals(0, heap.size);

        assertTrue(heap.offer(3, 1.0f));
        assertTrue(heap.offer(4, 2.0f));
        assertEquals(2, heap.size);
    }

    @Test
    public void popMinIntoExtractsDescendingOrder() {
        var heap = new VectorSearch.TopKHeap(3);

        assertTrue(heap.offer(101, 0.5f));
        assertTrue(heap.offer(102, 0.9f));
        assertTrue(heap.offer(103, 0.7f));
        assertFalse(heap.offer(104, 0.6f));
        assertEquals(3, heap.size);
        assertEquals(0.6f, heap.minScore());

        var ids = new int[3];
        var scores = new float[3];

        heap.popMinInto(ids, scores, 2);
        heap.popMinInto(ids, scores, 1);
        heap.popMinInto(ids, scores, 0);

        assertArrayEquals(new int[]{102, 103, 104}, ids);
        assertArrayEquals(new float[]{0.9f, 0.7f, 0.6f}, scores, 1e-6f);
        assertEquals(0, heap.size);
    }

    @ParameterizedTest
    @EnumSource(Dtype.class)
    void fullScanReturnsTopMatches(Dtype dtype, @TempDir Path tempDir) throws Exception {
        int dim = 128;
        long shardSizeBytes = (long) dim * dtype.bytes() * 2;

        float[][] rows = new float[3][dim];
        rows[0][0] = 1.0f;
        rows[1][1] = 1.0f;
        rows[2][1] = 0.6f;
        rows[2][2] = 0.8f;

        try (VectorMemoryWriter writer = VectorMemoryWriter.create(tempDir, dim, dtype, shardSizeBytes)) {
            for (int i = 0; i < rows.length; i++) {
                var row = writer.allocRow(i);
                row.putArray(rows[i], 0);
            }
        }

        VectorLayout layout = VectorLayout.of(tempDir, dim, dtype, shardSizeBytes);
        try (VectorMemoryReader reader = new VectorMemoryReader(layout, rows.length)) {
            VectorSearch search = new VectorSearch(reader);
            float[] query = rows[2].clone();
            VectorMath.normalize(query);

            VectorSearch.DocScore[] hits = search.similaritySearchFullScan(query, 2);
            assertEquals(2, hits.length);
            assertEquals(2, hits[0].docId());
            assertEquals(1, hits[1].docId());
            assertTrue(hits[0].score() >= hits[1].score());
            assertTrue(hits[0].score() > 0.9f);
        }
    }
}
