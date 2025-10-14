package io.github.mboros1.vs4j.core.vectors.search;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;
import io.github.mboros1.vs4j.core.vectors.memory.reader.VectorMemoryReader;

import java.util.concurrent.ForkJoinPool;



public class VectorSearch {
    private final VectorMemoryReader reader;
    private final ForkJoinPool pool;

    public VectorSearch(VectorMemoryReader reader) {
        this.reader = reader;
        pool = new ForkJoinPool();
    }


    public DocScore[] similaritySearchFullScan(float[] vector, int topK) {
        SearchCtx ctx = new SearchCtx(topK, reader.dim(), reader.dtype());
        try {

        } finally {
            ctx.close();
        }
        return null;
    }

    public record DocScore(int docId, float score) {}

    final static class TopKHeap {
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
                s[size]=score;
                id[size]=docId;
                siftUp(size++);
                return true;
            }

            if (score <= s[0]) return false;

            s[0]=score;
            id[0]=docId;
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
            outId[idx]    = id[0];
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
            float f =  s[a];  s[a] =  s[b];  s[b] = f;
            int  i = id[a]; id[a] = id[b]; id[b] = i;
        }
    }


    private final static class SearchCtx implements AutoCloseable {
        final int topK;
        final Dtype dtype;
        final ThreadLocal<TopKHeap> tlHeap;
        final ThreadLocal<float[]> tlTile;

        SearchCtx(int tile, int topK, Dtype dtype) {
            this.topK = topK;
            this.dtype = dtype;
            tlHeap = ThreadLocal.withInitial(() -> new TopKHeap(topK));
            tlTile = ThreadLocal.withInitial(() -> new float[tile]);
        }

        @Override public void close() { tlHeap.remove(); tlTile.remove(); }
    }
}
