import java.util.Scanner;
import java.util.function.Function;

public class TestMethods {
    private static final double LEARNING_RATE = 0.01;
    private static final double NUM_OF_REPETITION = 20;

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        Function<Double, Double> function = x -> Math.pow(x, 6) + Math.sin(x);
        Function<Double, Double> gradient = x -> 6 * Math.pow(x, 5) + Math.cos(x);

        double result = runSequentialGD(gradient, LEARNING_RATE, 10, 0.001);

        System.out.println("Function: f(x) = x^6 + sin(x)");
        System.out.println("Global minimum Sequential: " + function.apply(result));

        System.out.println("\nCalculation of average time of methods");
        int iterations = Main.getIterationsFromKeyboard(scanner);
        int threads = Main.getThreadsFromKeyboard(scanner);
        getAverageTimeByTwentyRep(iterations, threads);
    }

    private static double runSequentialGD(Function<Double, Double> gradient, double alpha, int maxIterations, double tolerance) {
        double x = 0.0;
        int iterations = 0;

        while (iterations < maxIterations) {
            double gradientValue = gradient.apply(x);
            double newX = x - alpha * gradientValue;
            if (Math.abs(newX - x) < tolerance) {
                break;
            }
            x = newX;
            iterations++;
        }

        return x;
    }

    private static void getAverageTimeByTwentyRep(int iterations, int threads) {
        double seqTime = 0;
        double parallTime = 0;
        double optimTime = 0;
        double[] parameters = Main.generateInitialParameters();

        for (int i = 0; i < NUM_OF_REPETITION; i++) {

            double startTime = System.nanoTime();
            Main.runSequentialGradientDescent(parameters, iterations);
            double endTime = System.nanoTime();

            double executionTime = (endTime - startTime) / 1e6;
            seqTime += executionTime;
        }

        for (int i = 0; i < NUM_OF_REPETITION; i++) {

            double startTime = System.nanoTime();
            Main.runParallelGradientDescent(parameters, iterations, threads);
            double endTime = System.nanoTime();

            double executionTime = (endTime - startTime) / 1e6;
            parallTime += executionTime;
        }

        for (int i = 0; i < NUM_OF_REPETITION; i++) {

            double startTime = System.nanoTime();
            Main.runOptimizedParallelGradientDescent(parameters, iterations, threads);
            double endTime = System.nanoTime();

            double executionTime = (endTime - startTime) / 1e6;
            optimTime += executionTime;
        }


        double averageParallelTime = parallTime / NUM_OF_REPETITION;
        double averageOptimizedTime = optimTime / NUM_OF_REPETITION;

        System.out.println("\nAverage time for Sequential GD - " + seqTime / NUM_OF_REPETITION);
        System.out.println("Average time for Parallel GD - " + parallTime / NUM_OF_REPETITION);
        System.out.println("Average time for Optimized GD - " + optimTime / NUM_OF_REPETITION);
        System.out.println("Average Acceleration - " + averageParallelTime / averageOptimizedTime);
    }
}