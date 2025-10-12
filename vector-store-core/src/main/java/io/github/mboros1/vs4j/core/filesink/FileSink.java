package io.github.mboros1.vs4j.core.filesink;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

public abstract class FileSink implements AutoCloseable {
    protected final FileChannel ch;
    protected final Path path;
    protected final ByteBuffer buf;        // direct, little-endian


    public FileSink(Path path, int bufferSize) throws IOException {
        this.ch = FileChannel.open(path, StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING, StandardOpenOption.WRITE);
        this.path = path;
        this.buf = ByteBuffer.allocateDirect(bufferSize).order(ByteOrder.LITTLE_ENDIAN);
    }

    public void flush() throws IOException {
        buf.flip();
        while (buf.hasRemaining()) ch.write(buf);
        buf.clear();
    }

    @Override
    public void close() throws Exception {
        flush();
        ch.close();
    }
}
