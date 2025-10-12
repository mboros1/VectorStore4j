package io.github.mboros1.vs4j.core.vectors.memory;

import com.fasterxml.jackson.core.JsonParser;

import java.io.IOException;
import java.nio.FloatBuffer;

public sealed interface RowCursor permits RowF32, RowF16 {
    int rowIndex();
    void putFromJsonArray(JsonParser jp) throws IOException;
    void putArray(float[] src, int off);
    void putBuffer(FloatBuffer fb);
}

