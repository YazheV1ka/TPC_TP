package test;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class MainTest {
    int iterations = 10000;
    int threads = 5;
    double eps = 0.001;
    double eps_optimized = 0.0001;

    @Test
    void testSequentialGradientDescentEPS() {
        double[] parameters = {1, 1, 1};
        double result = TestMethods.runSequentialGradientDescentAccyrancy(parameters, iterations);
        assertTrue(result < eps, "Result is not within the acceptable range");
    }

    @Test
    void testParallelStochasticGradientDescentEPS() {
        double[] parameters = {1, 1, 1};
        double result = TestMethods.runParallelStochasticGradientDescentAccyrancy(parameters, iterations, threads);
        assertTrue(result < eps, "Result is not within the acceptable range");
    }

    @Test
    void testOptimizedParallelGradientDescentEPS() {
        double[] parameters = {1, 1, 1};
        double result = TestMethods.runOptimizedParallelGradientDescentAccyrancy(parameters, iterations, threads);
        assertTrue(result < eps, "Result is not within the acceptable range");
    }

    @Test
    void testSequentialGradientDescentEPSOptimized() {
        //false
        double[] parameters = {1, 1, 1};
        double result = TestMethods.runSequentialGradientDescentAccyrancy(parameters, iterations);
        assertTrue(result < eps_optimized, "Result is not within the acceptable range");
    }

    @Test
    void testParallelStochasticGradientDescentEPSOptimized() {
        //false
        double[] parameters = {1, 1, 1};
        double result = TestMethods.runParallelStochasticGradientDescentAccyrancy(parameters, iterations, threads);
        assertTrue(result < eps_optimized, "Result is not within the acceptable range");
    }

    @Test
    void testOptimizedParallelGradientDescentEPSOptimized() {
        //true
        double[] parameters = {1, 1, 1};
        double result = TestMethods.runOptimizedParallelGradientDescentAccyrancy(parameters, iterations, threads);
        assertTrue(result < eps_optimized, "Result is not within the acceptable range");
    }
}
