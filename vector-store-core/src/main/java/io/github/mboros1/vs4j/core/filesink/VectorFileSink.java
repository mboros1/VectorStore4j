package io.github.mboros1.vs4j.core.filesink;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicInteger;

public final class VectorFileSink extends FileSink {
    private final int dim;
    private final int align = 64;
    private final int rowBytes = align * Float.BYTES;
    private final int buffSize;
    private final Object writeLock = new Object();
    private final AtomicInteger docIdNext = new AtomicInteger(0);
    private final AtomicInteger docIdWritten = new AtomicInteger(0);

    /**
     * Ctor.
     *
     * @param path of the binary vector file to be written
     * @param dim of the vector
     * @param ioBufferFloats size of IO Buffer in floats; recommended to be a multiple of the vector dim
     * @throws IOException if there's an issue creating a file channel for the given `path`
     */
    public VectorFileSink(Path path, int dim, int ioBufferFloats) throws IOException {
        super(path, ioBufferFloats * Float.BYTES);
        this.dim = dim;
        this.buffSize = ioBufferFloats;
    }

    /**
     * At the start of the JSON "embeddings" array, assumes `dim` elements in the array, throws exception if not.
     *
     * @param jsonParser that is on the `START_ARRAY` token of the embedding array
     */
    public void writeVectorFromJson(JsonParser jsonParser) throws IOException {
        if (jsonParser.currentToken() != JsonToken.START_ARRAY) throw new IOException("expected START_ARRAY");
        synchronized (writeLock) {
            if (buf.remaining() < dim * Float.BYTES) flush();
            int remaining = buf.remaining() / Float.BYTES;
            while (jsonParser.nextToken() != JsonToken.END_ARRAY) {
                var f = jsonParser.getFloatValue();
                buf.putFloat(f);
            }
            int curr = buf.remaining() / Float.BYTES;
            if (curr - remaining != dim * Float.BYTES) throw new IllegalStateException("Illegal embedding size: "
                    + (curr-remaining) + ", expected: " + dim);
        }
    }
}

