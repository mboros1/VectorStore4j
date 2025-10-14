package io.github.mboros1.vs4j.core.vectors.search;

import org.junit.jupiter.api.Test;

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
}
