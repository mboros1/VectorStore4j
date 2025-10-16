package io.github.mboros1.vs4j.core.vectors.search;

import io.github.mboros1.vs4j.core.vectors.memory.reader.VectorMemoryReader;

import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveAction;

public class VectorSearch {
    private final VectorMemoryReader reader;
    private final ForkJoinPool pool;

    public VectorSearch(VectorMemoryReader reader) {
        this.reader = Objects.requireNonNull(reader, "reader");
        this.pool = new ForkJoinPool(Math.max(1, Runtime.getRuntime().availableProcessors()));
    }

    public DocScore[] similaritySearchFullScan(float[] query, int topK) {
        Objects.requireNonNull(query, "query");
        if (topK <= 0) throw new IllegalArgumentException("topK must be positive");

        int dim = reader.dim();
        if (query.length != dim) {
            throw new IllegalArgumentException(STR."query dim mismatch: expected \{dim} got \{query.length}");
        }
        if (reader.numDocs() == 0) {
            return new DocScore[0];
        }

        int rowsPerSpan = reader.suggestRowsPerSpanKiB(1024);
        List<VectorMemoryReader.Span> spans = reader.spans(rowsPerSpan);
        if (spans.isEmpty()) {
            return new DocScore[0];
        }

        ConcurrentLinkedQueue<TopKHeap> partials = new ConcurrentLinkedQueue<>();
        pool.invoke(new SpanSearchTask(spans, 0, spans.size(), query, topK, partials));

        TopKHeap merged = new TopKHeap(topK);
        for (TopKHeap partial : partials) {
            int size = partial.size;
            for (int i = 0; i < size; i++) {
                merged.offer(partial.id[i], partial.s[i]);
            }
        }

        return drainTopK(merged, topK);
    }

    public record DocScore(int docId, float score) {}

    static final class TopKHeap {
        final float[] s;
        final int[] id;
        int size;

        TopKHeap(int k) {
            if (k <= 0) throw new IllegalArgumentException(STR."K must be positive, k:\{k}");
            s = new float[k];
            id = new int[k];
            size = 0;
        }

        boolean offer(int docId, float score) {
            if (!Float.isFinite(score)) return false;

            if (size < s.length) {
                s[size] = score;
                id[size] = docId;
                siftUp(size++);
                return true;
            }

            if (score <= s[0]) return false;

            s[0] = score;
            id[0] = docId;
            siftDown(0);
            return false;
        }

        float minScore() { return s[0]; }

        int popMinId() {
            int i = id[0];
            swap(0, --size);
            if (size > 0) siftDown(0);
            return i;
        }

        void popMinInto(int[] outId, float[] outScore, int idx) {
            outId[idx] = id[0];
            outScore[idx] = s[0];
            swap(0, --size);
            if (size > 0) siftDown(0);
        }

        private void siftUp(int i) {
            while (i > 0) {
                int p = (i - 1) >>> 1;
                if (s[i] >= s[p]) break;
                swap(i, p);
                i = p;
            }
        }

        private void siftDown(int i) {
            int half = size >>> 1;
            while (i < half) {
                int left = (i << 1) + 1;
                int right = left + 1;

                int smallest = left;
                if (right < size && s[right] < s[left]) {
                    smallest = right;
                }

                if (s[i] <= s[smallest]) break;
                swap(i, smallest);
                i = smallest;
            }
        }

        private void swap(int a, int b) {
            float f = s[a];  s[a] = s[b];  s[b] = f;
            int i = id[a];   id[a] = id[b]; id[b] = i;
        }
    }

    private static DocScore[] drainTopK(TopKHeap heap, int topK) {
        int n = Math.min(topK, heap.size);
        if (n <= 0) return new DocScore[0];

        int[] ids = new int[n];
        float[] scores = new float[n];
        for (int i = n - 1; i >= 0; i--) {
            heap.popMinInto(ids, scores, i);
        }

        DocScore[] out = new DocScore[n];
        for (int i = 0; i < n; i++) {
            out[i] = new DocScore(ids[i], scores[i]);
        }
        return out;
    }

    private static final class SpanSearchTask extends RecursiveAction {
        private static final int THRESHOLD = 8;

        private final List<VectorMemoryReader.Span> spans;
        private final int start;
        private final int end;
        private final float[] query;
        private final int topK;
        private final ConcurrentLinkedQueue<TopKHeap> partials;

        SpanSearchTask(List<VectorMemoryReader.Span> spans,
                       int start,
                       int end,
                       float[] query,
                       int topK,
                       ConcurrentLinkedQueue<TopKHeap> partials) {
            this.spans = spans;
            this.start = start;
            this.end = end;
            this.query = query;
            this.topK = topK;
            this.partials = partials;
        }

        @Override
        protected void compute() {
            int length = end - start;
            if (length <= THRESHOLD) {
                for (int i = start; i < end; i++) {
                    VectorMemoryReader.Span span = spans.get(i);
                    TopKHeap heap = new TopKHeap(topK);
                    span.bulkDot(query, (docId, score) -> heap.offer(docId, score));
                    if (heap.size > 0) {
                        partials.add(heap);
                    }
                }
                return;
            }

            int mid = start + (length >>> 1);
            SpanSearchTask left = new SpanSearchTask(spans, start, mid, query, topK, partials);
            SpanSearchTask right = new SpanSearchTask(spans, mid, end, query, topK, partials);
            left.fork();
            right.compute();
            left.join();
        }
    }
}
