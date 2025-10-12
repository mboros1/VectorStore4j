package io.github.mboros1.vs4j.core.vectors.enums;

public enum Dtype {
    F16,
    F32;

    public int bytes() {
        return this.name().equals("F16") ? 2 : 4;
    }
}
