package io.github.mboros1.vs4j.core.vectors.layout;

import io.github.mboros1.vs4j.core.vectors.enums.Dtype;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.nio.file.Path;

public class VectorLayoutTest {

    @ParameterizedTest
    @EnumSource(Dtype.class)
    public void testLayout(Dtype dtype, @TempDir Path tempDir) {
        var layout1 = VectorLayout.of(tempDir, 1024, dtype);
        var layout2 = VectorLayout.of(tempDir, 1024, dtype, 1024*1024);

        System.out.println(layout1);
        System.out.println(layout2);

        Assertions.assertEquals(layout1.rowStride(), layout1.rowBytes());
        Assertions.assertEquals(layout2.rowStride(), layout2.rowBytes());
    }
}
