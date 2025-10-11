package io.github.mboros1.vs4j.core.binformat.loader;

import com.fasterxml.jackson.core.JsonFactory;
import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.core.JsonToken;
import com.jsoniter.JsonIterator;
import com.jsoniter.ValueType;
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
    public static record DocDigest(long hash64, int textChars) {
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
                .forEach(bb -> ingestDocuments(bb, digests));

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

    private static void ingestDocumentsJsonIter(InputStream in, Map<String, DocDigest> digests) throws IOException {
        JsonIterator it = JsonIterator.parse(in, 1 << 14); // 16KiB buffer
        if (!it.readArray()) {
            throw new IllegalArgumentException("Top-level array expected: ");
        }
        do {
            // element object
            if (it.whatIsNext() != ValueType.OBJECT) {
                it.skip(); // be tolerant
                continue;
            }
            String filename = null;
            long hash64 = 0L;
            int lenChars = 0;

            for (String field = it.readObject(); field != null; field = it.readObject()) {
                switch (field) {
                    case "meta" -> {
                        if (it.whatIsNext() == ValueType.OBJECT) {
                            for (String m = it.readObject(); m != null; m = it.readObject()) {
                                if ("origin".equals(m) && it.whatIsNext() == ValueType.OBJECT) {
                                    for (String o = it.readObject(); o != null; o = it.readObject()) {
                                        if ("filename".equals(o)) {
                                            filename = it.readString();
                                        } else {
                                            it.skip();
                                        }
                                    }
                                } else {
                                    it.skip();
                                }
                            }
                        } else {
                            it.skip();
                        }
                    }
                    case "text" -> {
                        String text = it.readString(); // fully materialize this field
                        hash64 = text.hashCode();
                        lenChars = text.length();
                    }
                    default -> it.skip();
                }
            }

            if (filename != null) {
                digests.put(filename, new DocDigest(hash64, lenChars));
            }
        } while (it.readArray());
    }

    private static void ingestDocuments(InputStream in, Map<String, DocDigest> digests) {
        try (JsonParser jp = JF.createParser(in)) {

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
                        case "meta" -> {
                            if (jp.currentToken() == JsonToken.START_OBJECT) {
                                // dive into meta → origin → filename
                                while (jp.nextToken() != JsonToken.END_OBJECT) {
                                    if (jp.currentToken() != JsonToken.FIELD_NAME) continue;
                                    String mfield = jp.getCurrentName();
                                    jp.nextToken();
                                    if ("origin".equals(mfield) && jp.currentToken() == JsonToken.START_OBJECT) {
                                        while (jp.nextToken() != JsonToken.END_OBJECT) {
                                            if (jp.currentToken() != JsonToken.FIELD_NAME) continue;
                                            String ofield = jp.getCurrentName();
                                            jp.nextToken();
                                            if ("filename".equals(ofield) && !jp.currentToken().isStructStart()) {
                                                filename = jp.getValueAsString(null);
                                            } else {
                                                jp.skipChildren();
                                            }
                                        }
                                    } else {
                                        jp.skipChildren();
                                    }
                                }
                            } else {
                                jp.skipChildren();
                            }
                        }
                        case "text" -> {
                            // Force full decode; then hash UTF-8 bytes (one alloc for bytes)
                            String text = jp.getValueAsString("");
                            byte[] utf8 = text.getBytes(StandardCharsets.UTF_8);
                            hash64 = text.hashCode();
                            lenChars = text.length();
                        }
                        default -> jp.skipChildren();
                    }
                }

                if (filename != null) {
                    digests.put(filename, new DocDigest(hash64, lenChars));
                }
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }

    public static Map<String, DocDigest> loadDirNoMmap(String dirName) {
        var file = new File(dirName);
        Map<String, DocDigest> digests = new HashMap<>();
        if (!file.isDirectory()) {
            log.warn("'{}' is not a directory, cannot load", dirName);
            return digests;
        }
        Arrays.stream(Objects.requireNonNull(file.listFiles())).parallel()
                .map(p -> {
                    try {
                        return new BufferedInputStream(new FileInputStream(p));
                    } catch (FileNotFoundException e) {
                        throw new RuntimeException(e);
                    }
                })
                .forEach(bb -> {
                    try {
                        ingestDocumentsJsonIter(bb, digests);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });
        return digests;
    }

    public static Map<String, DocDigest> loadDir(String dirName) {
        var file = new File(dirName);
        Map<String, DocDigest> digests = new HashMap<>();
        if (!file.isDirectory()) {
            log.warn("'{}' is not a directory, cannot load", dirName);
            return digests;
        }
        Arrays.stream(Objects.requireNonNull(file.listFiles())).parallel()
                .map(jsonFile -> Path.of(jsonFile.toURI()))
                .map(JsonLoader::mmapReadOnly)
                .forEach(bb -> {
                    try {
                        ingestDocumentsJsonIter(bb, digests);
                    } catch (IOException e) {
                        throw new RuntimeException(e);
                    }
                });
        return digests;
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

    static int countTopLevelArrayElements(InputStream in) {
        try (JsonParser p = JF.createParser(in)) {

            JsonToken t = p.nextToken();
            if (t != JsonToken.START_ARRAY) {
                throw new IllegalArgumentException("Expected top-level array in");
            }

            int count = 0;
            // Move to first element or END_ARRAY
            while ((t = p.nextToken()) != JsonToken.END_ARRAY) {
                // We’re positioned on the first token of an element (object/array/scalar).
                count++;
                p.skipChildren(); // fast-forward over the element payload
            }
            return count;

        } catch (IOException e) {
            throw new UncheckedIOException("Failed parsing", e);
        }
    }
}
