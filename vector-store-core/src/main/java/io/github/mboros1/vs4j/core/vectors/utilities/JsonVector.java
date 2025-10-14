package io.github.mboros1.vs4j.core.vectors.utilities;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;

import java.io.IOException;

public class JsonVector {
    public static void parseVectorFromJson(JsonParser jp, float[] buf) throws IOException {
        if (jp.currentToken() != JsonToken.START_ARRAY)
            throw new IOException("expected START_ARRAY");
        int i = 0;
        while (jp.nextToken() != JsonToken.END_ARRAY) {
            float v;
            if (jp.currentToken().isNumeric()) {
                v = jp.getFloatValue();
            } else if (jp.currentToken() == JsonToken.VALUE_STRING) {
                v = Float.parseFloat(jp.getValueAsString());
            } else {
                throw new IOException(STR."non-numeric value in embedding: \{jp.currentToken()}");
            }
            buf[i++] = v;
        }
        if (i != buf.length) throw new IOException(STR."JSON array dim does not match expected, JSON array: \{i}, expected: \{buf.length}");
    }
}
