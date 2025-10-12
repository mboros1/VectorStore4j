package io.github.mboros1.vs4j.core.loader;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import io.github.mboros1.vs4j.core.filesink.VectorFileSink;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.*;

import static java.nio.file.StandardOpenOption.READ;

public class JsonLoader {
    public record DocDigest(long hash64, int textChars) {
    }

    private static final Logger log = LoggerFactory.getLogger(JsonLoader.class);
    private static final JsonFactory JF = new JsonFactory();

    private JsonLoader() {
    }

    public static Map<String, DocDigest> loadDirJackson(String dirName) {
        var file = new File(dirName);
        var digests = new HashMap<String, DocDigest>();
        if (!file.isDirectory()) {
            log.warn("'{}' is not a directory, cannot load", dirName);
            return digests;
        }
        Arrays.stream(Objects.requireNonNull(file.listFiles())).parallel()
                .map(jsonFile -> Path.of(jsonFile.toURI()))
                .map(JsonLoader::mmapReadOnly)
                .forEach(JsonLoader::ingestDocuments);

        return digests;
    }

    public static Map<String, DocDigest> ingestDocumentsNoMmap(String dirName) throws IOException {
        Path dir = Path.of(dirName);
        Map<String, DocDigest> digests = new HashMap<>();
        Arrays.stream(Objects.requireNonNull(dir.toFile().listFiles()))
                .parallel()
                .forEach(p -> {
                    try {
                        ingestDocuments(new BufferedInputStream(new FileInputStream(p)), digests);
                    } catch (FileNotFoundException e) {
                        throw new RuntimeException(e);
                    }
                });
        return digests;
    }

    private static void ingestDocuments(InputStream in) {
        try (JsonParser jp = JF.createParser(in); VectorFileSink sink = new VectorFileSink("", 1024, 10*1024)) {

            if (jp.nextToken() != JsonToken.START_ARRAY) {
                throw new IllegalArgumentException("Top-level array expected: ");
            }
            while (jp.nextToken() != JsonToken.END_ARRAY) {
                // Parse one element: { meta: {...}, text: "..." }
                String filename = null;
                long hash64 = 0L;
                int lenChars = 0;

                if (jp.currentToken() != JsonToken.START_OBJECT) {
                    throw new IllegalStateException("Array element must be object: ");
                }
                while (jp.nextToken() != JsonToken.END_OBJECT) {
                    if (jp.currentToken() != JsonToken.FIELD_NAME) continue;
                    String field = jp.getCurrentName();
                    jp.nextToken();
                    switch (field) {
                        case "meta", "metadata" -> {
                            if (jp.currentToken() == JsonToken.START_OBJECT) {
                                // dive into meta → origin → filename
                                while (jp.nextToken() != JsonToken.END_OBJECT) {
                                    if (jp.currentToken() != JsonToken.FIELD_NAME) continue;
                                    String mfield = jp.getCurrentName();
                                    jp.nextToken();
                                    if (mfield.equals("embeddings") && jp.currentToken() == JsonToken.START_ARRAY) {

                                    }
                                }
                            } else {
                                jp.skipChildren();
                            }
                        }
                        default -> jp.skipChildren();
                    }
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    private static ByteBufferInputStream mmapReadOnly(Path path) {
        try (FileChannel fc = FileChannel.open(path, READ)) {
            long size = fc.size();
            var bb = fc.map(FileChannel.MapMode.READ_ONLY, 0, size)
                    .order(ByteOrder.LITTLE_ENDIAN);
            return new ByteBufferInputStream(bb);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    static final class ByteBufferInputStream extends InputStream {
        private final ByteBuffer buf;

        ByteBufferInputStream(ByteBuffer buf) {
            this.buf = buf.slice();
        }

        @Override
        public int read() {
            return buf.hasRemaining() ? (buf.get() & 0xFF) : -1;
        }

        @Override
        public int read(byte[] b, int off, int len) {
            if (!buf.hasRemaining()) return -1;
            int n = Math.min(len, buf.remaining());
            buf.get(b, off, n);
            return n;
        }
    }
}
