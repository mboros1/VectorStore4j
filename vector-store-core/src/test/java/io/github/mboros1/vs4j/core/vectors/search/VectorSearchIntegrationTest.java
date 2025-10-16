package io.github.mboros1.vs4j.core.vectors.search;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import io.github.mboros1.vs4j.core.vectors.enums.Dtype;
import io.github.mboros1.vs4j.core.vectors.memory.layout.VectorLayout;
import io.github.mboros1.vs4j.core.vectors.memory.reader.VectorMemoryReader;
import io.github.mboros1.vs4j.core.vectors.memory.writer.VectorMemoryWriter;
import io.github.mboros1.vs4j.core.vectors.utilities.JsonVector;
import io.github.mboros1.vs4j.core.vectors.utilities.VectorMath;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ThreadLocalRandom;

import static org.junit.jupiter.api.Assertions.*;

class VectorSearchIntegrationTest {
    private static final JsonFactory JSON_FACTORY = new JsonFactory();

    @Test
    void embeddedSamplesYieldRelevantHits(@TempDir Path tempDir) throws Exception {
        Path embeddedDir = Paths.get("..", "samples", "embedded").toAbsolutePath().normalize();
        assertTrue(Files.isDirectory(embeddedDir), "embedded samples directory missing");

        List<Path> files;
        try (var stream = Files.list(embeddedDir)) {
            files = stream.filter(Files::isRegularFile)
                    .sorted(Comparator.comparing(Path::getFileName))
                    .toList();
        }
        assertFalse(files.isEmpty(), "no embedded sample files discovered");

        int dim = detectDim(files.get(0));

        var expectations = List.of(
                new QueryExpectation(
                        "Which organization emphasises translational research for the Mexican population?",
                        "mexican institute for social security"
                ),
                new QueryExpectation(
                        "Which neural pathways coordinate locomotion under visually guided conditions?",
                        "striatal output neurons"
                ),
                new QueryExpectation(
                        "Which autoimmune therapy is discussed alongside cutaneous tuberculosis care?",
                        "hydroxychloroquine"
                )
        );

        var rowTexts = new ArrayList<String>();
        int nextRowId = 0;

        try (VectorMemoryWriter writer = VectorMemoryWriter.create(tempDir, dim, Dtype.F32)) {
            for (Path file : files) {
                nextRowId = ingestFile(file, writer, dim, rowTexts, expectations, nextRowId);
            }
        }

        assertFalse(rowTexts.isEmpty(), "no chunks ingested from samples");
        assertEquals(nextRowId, rowTexts.size(), "row count mismatch");
        for (QueryExpectation expectation : expectations) {
            assertTrue(expectation.isCaptured(), "failed to locate evidence for: " + expectation.question());
        }

        VectorLayout layout = VectorLayout.of(tempDir, dim, Dtype.F32);
        try (VectorMemoryReader reader = new VectorMemoryReader(layout, rowTexts.size())) {
            VectorSearch search = new VectorSearch(reader);

            for (QueryExpectation expectation : expectations) {
                VectorSearch.DocScore[] hits = search.similaritySearchFullScan(expectation.queryVector(), 3);
                assertTrue(hits.length > 0, "expected matches for " + expectation.question());
                assertEquals(expectation.rowId(), hits[0].docId(), "seed chunk should rank first: " + expectation.question());

                boolean matched = false;
                for (VectorSearch.DocScore hit : hits) {
                    String text = rowTexts.get(hit.docId()).toLowerCase(Locale.ROOT);
                    if (text.contains(expectation.matchPhrase())) {
                        matched = true;
                        break;
                    }
                }
                assertTrue(matched, "top hits should reference '" + expectation.matchPhrase() + "'");
            }

            benchmarkQueries(search, expectations, rowTexts.size(), reader.numDocs());
        }
    }

    private static int ingestFile(Path file,
                                  VectorMemoryWriter writer,
                                  int dim,
                                  ArrayList<String> rowTexts,
                                  List<QueryExpectation> expectations,
                                  int nextRowId) throws IOException {
        try (JsonParser parser = JSON_FACTORY.createParser(file.toFile())) {
            if (parser.nextToken() != JsonToken.START_ARRAY) {
                throw new IOException("expected START_ARRAY in " + file);
            }
            while (parser.nextToken() != JsonToken.END_ARRAY) {
                if (parser.currentToken() != JsonToken.START_OBJECT) {
                    parser.skipChildren();
                    continue;
                }
                String chunkText = null;
                float[] embedding = null;

                while (parser.nextToken() != JsonToken.END_OBJECT) {
                    String field = parser.getCurrentName();
                    parser.nextToken();
                    if ("text".equals(field)) {
                        chunkText = parser.getValueAsString();
                    } else if ("metadata".equals(field) && parser.currentToken() == JsonToken.START_OBJECT) {
                        while (parser.nextToken() != JsonToken.END_OBJECT) {
                            String metaField = parser.getCurrentName();
                            parser.nextToken();
                            if ("embedding".equals(metaField)) {
                                embedding = new float[dim];
                                JsonVector.parseVectorFromJson(parser, embedding);
                            } else {
                                parser.skipChildren();
                            }
                        }
                    } else {
                        parser.skipChildren();
                    }
                }

                if (chunkText == null || embedding == null) {
                    continue; // skip malformed chunk
                }

                int rowId = nextRowId++;
                var row = writer.allocRow(rowId);
                row.putArray(embedding, 0);

                rowTexts.add(chunkText);
                String lower = chunkText.toLowerCase(Locale.ROOT);
                for (QueryExpectation expectation : expectations) {
                    expectation.tryCapture(lower, embedding, rowId);
                }
            }
        }
        return nextRowId;
    }

    private static void benchmarkQueries(VectorSearch search,
                                         List<QueryExpectation> expectations,
                                         int rowsInCorpus,
                                         int readerRows) {
        int queries = 10_000;
        int topK = 3;

        long start = System.nanoTime();
        ThreadLocalRandom rnd = ThreadLocalRandom.current();
        for (int i = 0; i < queries; i++) {
            QueryExpectation expected = expectations.get(rnd.nextInt(expectations.size()));
            search.similaritySearchFullScan(expected.queryVector(), topK);
        }
        long elapsedNs = System.nanoTime() - start;
        double avgUs = elapsedNs / (double) queries / 1_000.0;

        System.out.printf(Locale.ROOT,
                "Benchmark: %,d queries, topK=%d, rows=%,d (reader reports %,d) -> avg %.3f µs%n",
                queries, topK, rowsInCorpus, readerRows, avgUs);
    }

    private static int detectDim(Path file) throws IOException {
        try (JsonParser parser = JSON_FACTORY.createParser(file.toFile())) {
            if (parser.nextToken() != JsonToken.START_ARRAY) {
                throw new IOException("expected START_ARRAY in " + file);
            }
            while (parser.nextToken() != JsonToken.END_ARRAY) {
                if (parser.currentToken() != JsonToken.START_OBJECT) {
                    parser.skipChildren();
                    continue;
                }
                while (parser.nextToken() != JsonToken.END_OBJECT) {
                    String field = parser.getCurrentName();
                    parser.nextToken();
                    if ("metadata".equals(field) && parser.currentToken() == JsonToken.START_OBJECT) {
                        while (parser.nextToken() != JsonToken.END_OBJECT) {
                            String metaField = parser.getCurrentName();
                            parser.nextToken();
                            if ("embedding".equals(metaField)) {
                                int dim = 0;
                                if (parser.currentToken() != JsonToken.START_ARRAY) {
                                    throw new IOException("expected START_ARRAY for embedding in " + file);
                                }
                                while (parser.nextToken() != JsonToken.END_ARRAY) {
                                    dim++;
                                }
                                if (dim == 0) throw new IOException("empty embedding in " + file);
                                return dim;
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
        throw new IOException("no embedding found in " + file);
    }

    private static final class QueryExpectation {
        private final String question;
        private final String matchPhrase;
        private float[] queryVector;
        private int rowId = -1;

        QueryExpectation(String question, String matchPhrase) {
            this.question = question;
            this.matchPhrase = matchPhrase.toLowerCase(Locale.ROOT);
        }

        void tryCapture(String chunkLower, float[] embedding, int rowId) {
            if (this.rowId != -1) return;
            if (!chunkLower.contains(matchPhrase)) return;

            float[] copy = embedding.clone();
            VectorMath.normalize(copy);
            this.queryVector = copy;
            this.rowId = rowId;
        }

        boolean isCaptured() {
            return rowId != -1 && queryVector != null;
        }

        String question() { return question; }
        String matchPhrase() { return matchPhrase; }
        int rowId() { return rowId; }
        float[] queryVector() { return queryVector; }
    }
}
