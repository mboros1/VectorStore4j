package io.github.mboros1.vs4j.core.orchestrator;

import java.util.concurrent.atomic.AtomicInteger;

public class Orchestrator {
    private final AtomicInteger docId = new AtomicInteger(0);

    public int nextDocId() {
        return docId.getAndIncrement();
    }
}
