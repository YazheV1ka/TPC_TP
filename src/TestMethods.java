import java.math.BigDecimal;
import java.util.Scanner;
import java.util.function.Function;

public class TestMethods {
    private static final double LEARNING_RATE = 0.1;
    private static final double eps = 0.001;
    private static final double NUM_OF_REPETITION = 20;

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        Function<Double, Double> function = x -> Math.pow(x, 6) + Math.sin(x);
        Function<Double, Double> gradient = x -> 6 * Math.pow(x, 5) + Math.cos(x);

        double resultGD = runSequentialGD(gradient);
        System.out.println("Function: f(x) = x^6 + sin(x)");
        System.out.println("LEARNING_RATE = 0.1");
        System.out.println("epsilon = 0.001");

        System.out.println("Local minimum Sequential GD: " + function.apply(resultGD));
        System.out.println("Minimum point Sequential GD: " + resultGD);

        System.out.println("\nCalculation of Dichotomy Method:");
        double resultDM = runDichotomyMethod(gradient);
        System.out.println("Local minimum Dichotomy Method: " + function.apply(resultDM));
        System.out.println("Minimum point Dichotomy Method: " + resultDM);

        System.out.println("\nCalculation of average time of methods");
        int iterations = Main.getIterationsFromKeyboard(scanner);
        int threads = Main.getThreadsFromKeyboard(scanner);
        getAverageTimeByTwentyRep(iterations, threads);
    }

    private static double runSequentialGD(Function<Double, Double> gradient) {
        double res = 0;
        for (int i = 0; i < 10; i++) {
            double gradientValue = gradient.apply(res);
            res -= LEARNING_RATE * gradientValue;
            if (Math.abs(res) < eps) {
                break;
            }
        }

        return res;
    }

    public static double runDichotomyMethod(Function<Double, Double> gradient) {
        double c = 0;
        double a = -1.0;
        double b = 0.0;
        int iterations = 10;

        BigDecimal epsilon = BigDecimal.valueOf(eps);

        for (int i = 1; i <= iterations; i++) {
            c = (a + b) / 2;
            BigDecimal fa = BigDecimal.valueOf(gradient.apply(a));
            BigDecimal fb = BigDecimal.valueOf(gradient.apply(b));
            BigDecimal fc = BigDecimal.valueOf(gradient.apply(c));

            System.out.println("Iteration " + i + ":");
            System.out.println("a" + i + " = " + a + ", b" + i + " = " + b + ", c" + i + " = " + c);
            System.out.println("f'(a" + i + ") = " + fa);
            System.out.println("f'(b" + i + ") = " + fb);
            System.out.println("f'(c" + i + ") = " + fc);

            if (fc.abs().compareTo(epsilon) < 0) {
                return c;
            } else if (fc.multiply(fa).compareTo(BigDecimal.ZERO) < 0) {
                b = c;
            } else {
                a = c;
            }

        }
        return c;
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