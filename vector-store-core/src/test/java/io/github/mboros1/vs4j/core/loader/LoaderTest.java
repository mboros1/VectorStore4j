package io.github.mboros1.vs4j.core.loader;

import io.github.mboros1.vs4j.core.binformat.loader.JsonLoader;
import org.junit.jupiter.api.Test;

import java.io.File;
import java.io.IOException;

public class LoaderTest {

    @Test
    void loadDir_samples_jackson() {
        var t0 = System.currentTimeMillis();
        var result = JsonLoader.loadDirJackson("../samples/");
        var t1 = System.currentTimeMillis();
        System.out.println(t1 - t0);
        System.out.println(result);
    }

    @Test
    void loadDir_samples_naive() throws IOException {
        var t0 = System.currentTimeMillis();
        var result = JsonLoader.ingestDocumentsNoMmap("../samples/");
        var t1 = System.currentTimeMillis();
        System.out.println(t1 - t0);
        System.out.println(result);
    }
}
