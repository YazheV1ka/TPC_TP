import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MainTest {
    int iterations = 10000;
    int threads = 5;
    double eps = 0.001;

    @Test
    void testSequentialGradientDescentEPS() {
        double[] parameters = {1, 1, 1};
        double result = Main.runSequentialGradientDescent(parameters, iterations);
        assertTrue(result < eps, "Result is not within the acceptable range");
    }

    @Test
    void testParallelStochasticGradientDescentEPS() {
        double[] parameters = {1, 1, 1};
        double result = Main.runParallelGradientDescent(parameters, iterations, threads);
        assertTrue(result < eps, "Result is not within the acceptable range");
    }

    @Test
    void testOptimizedParallelGradientDescentEPS() {
        double[] parameters = {1, 1, 1};
        double result = Main.runOptimizedParallelGradientDescent(parameters, iterations, threads);
        assertTrue(result < eps, "Result is not within the acceptable range");
    }

}
