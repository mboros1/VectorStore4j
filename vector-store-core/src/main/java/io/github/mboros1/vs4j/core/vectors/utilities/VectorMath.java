package io.github.mboros1.vs4j.core.vectors.utilities;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import java.util.Arrays;

public class VectorMath {
    private static final VectorSpecies<Float> SPEC = FloatVector.SPECIES_PREFERRED;

    public static void normalize(float[] vector) {
        final int n = vector.length;
        if (n == 0) return;

        double sumSq = 0.0;
        final int upper = SPEC.loopBound(n);

        // SIMD accumulate sum of squares
        int i = 0;
        for (; i < upper; i += SPEC.length()) {
            FloatVector v = FloatVector.fromArray(SPEC, vector, i);
            sumSq += v.mul(v).reduceLanes(VectorOperators.ADD);
        }
        // Tail
        for (; i < n; i++) {
            float x = vector[i];
            sumSq += (double) x * x;
        }

        if (sumSq == 0.0) { // zero vector → leave zeros
            Arrays.fill(vector, 0f);
            return;
        }

        final float inv = (float) (1.0 / Math.sqrt(sumSq));

        // SIMD scale in place
        i = 0;
        for (; i < upper; i += SPEC.length()) {
            FloatVector v = FloatVector.fromArray(SPEC, vector, i);
            v.mul(inv).intoArray(vector, i);
        }
        // Tail
        for (; i < n; i++) {
            vector[i] *= inv;
        }
    }
}
