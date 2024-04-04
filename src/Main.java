import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.*;

import java.util.concurrent.atomic.AtomicInteger;

public class Main {
    private static final double LEARNING_RATE = 0.001;
    private static final BigDecimal epsilon = BigDecimal.valueOf(0.001);

    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);

        int iterations = getIterationsFromKeyboard(scanner);
        int threads = getThreadsFromKeyboard(scanner);

        double[] parameters = generateInitialParameters();

        double startTimeSequentialMin = System.nanoTime();
        double resultSequentialMin = runSequentialGradientDescent(parameters, iterations);
        double endTimeSequentialMin = System.nanoTime();
        double timeSequential = (endTimeSequentialMin - startTimeSequentialMin) / 1e6;

        double startTimeStochasticMin = System.nanoTime();
        double resultStochasticMin = runParallelGradientDescent(parameters, iterations, threads);
        double endTimeStochasticMin = System.nanoTime();
        double timeStochastic = (endTimeStochasticMin - startTimeStochasticMin) / 1e6;

        double startTimeOptimizedMin = System.nanoTime();
        double resultOptimizedMin = runOptimizedParallelGradientDescent(parameters, iterations, threads);
        double endTimeOptimizedMin = System.nanoTime();
        double timeStochasticOptimized = (endTimeOptimizedMin - startTimeOptimizedMin) / 1e6;

        double accelerationStochasticOptimizedMin = timeStochastic / timeStochasticOptimized;

        System.out.println("\nSequential Descent Result: " + resultSequentialMin);
        System.out.println("Sequential Descent Execution time: " + timeSequential + " milliseconds\n");

        System.out.println("Parallel Descent Result: " + resultStochasticMin);
        System.out.println("Parallel Descent Execution time: " + timeStochastic + " milliseconds\n");

        System.out.println("Optimized Parallel Descent Result: " + resultOptimizedMin);
        System.out.println("Optimized Parallel Descent Execution time: " + timeStochasticOptimized + " milliseconds");
        System.out.println("Optimized Parallel Descent Acceleration: " + accelerationStochasticOptimizedMin);
    }


    public static int getIterationsFromKeyboard(Scanner scanner) {
        System.out.print("Enter the number of iterations: ");
        return scanner.nextInt();
    }

    public static int getThreadsFromKeyboard(Scanner scanner) {
        System.out.print("Enter the number of threads: ");
        return scanner.nextInt();
    }

    public static double[] generateInitialParameters() {
        return new double[]{1, 1, 1};
    }

    public static double runSequentialGradientDescent(double[] parameters, int iterations) {
        PreviousValues previousValues = new PreviousValues();
        for (int i = 0; i < iterations; i++) {
            double[] gradient = calculateGradient(parameters);
            for (int j = 0; j < parameters.length; j++) {
                parameters[j] -= LEARNING_RATE * gradient[j];
            }
            double cost = Main.calculateFunction(parameters);
            BigDecimal accuracy = BigDecimal.valueOf(Math.abs(cost - previousValues.getCost()));
            previousValues.setCost(cost);
            previousValues.setAccuracy(accuracy);
            System.out.println("[Sequential Gradient] Iteration " + i + ": Cost = " + previousValues.getCost() + ", Accuracy = " + previousValues.getAccuracy().toPlainString());

            if (accuracy.compareTo(epsilon) < 0) {
                break;
            }
        }
        return calculateFunction(parameters);
    }


    public static double runParallelGradientDescent(double[] parameters, int iterations, int threads) {
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        Queue<double[]> gradientQueue = new LinkedList<>();
        AtomicInteger iterationsCount = new AtomicInteger(0);
        PreviousValues previousValues = new PreviousValues();
        final Object lock = new Object();

        try {
            int iterationsPerThread = iterations / threads;

            for (int t = 0; t < threads; t++) {
                final int startIteration = t * iterationsPerThread;
                final int endIteration = (t + 1) * iterationsPerThread;

                executor.submit(() -> {
                    for (int i = startIteration; i < endIteration; i++) {
                        double[] gradient;
                        synchronized (lock) {
                            if (gradientQueue.isEmpty()) {
                                gradient = calculateGradient(parameters);
                            } else {
                                gradient = gradientQueue.poll();
                            }
                            for (int k = 0; k < parameters.length; k++) {
                                parameters[k] -= LEARNING_RATE * gradient[k];
                            }
                        }
                        iterationsCount.incrementAndGet();
                        double cost = calculateFunction(parameters);
                        BigDecimal accuracy = BigDecimal.valueOf(Math.abs(cost - previousValues.getCost()));
                        previousValues.setCost(cost);
                        previousValues.setAccuracy(accuracy);
                        System.out.println("[Parallel Gradient] Iteration " + i + ": Cost = " + previousValues.getCost() + ", Accuracy = " + previousValues.getAccuracy().toPlainString());

                        if (accuracy.compareTo(epsilon) < 0) {
                            break;
                        }
                    }
                });
            }

            executor.shutdown();
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
            return Double.NaN;
        }

        return calculateFunction(parameters);
    }


    public static double runOptimizedParallelGradientDescent(double[] parameters, int iterations, int threads) {
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        AtomicInteger iterationsCount = new AtomicInteger(0);
        PreviousValues previousValues = new PreviousValues();
        final Object lock = new Object();

        try {
            int iterationsPerThread = iterations / threads;

            for (int t = 0; t < threads; t++) {
                final int startIteration = t * iterationsPerThread;
                final int endIteration = (t + 1) * iterationsPerThread;

                executor.submit(() -> {
                    double[] localParameters = Arrays.copyOf(parameters, parameters.length);
                    for (int i = startIteration; i < endIteration; i++) {
                        double[] gradient;
                        synchronized (lock){
                            gradient = calculateGradient(localParameters);
                            for (int k = 0; k < parameters.length; k++) {
                                localParameters[k] -= LEARNING_RATE * gradient[k];
                            }
                        }

                        iterationsCount.incrementAndGet();
                        double cost = calculateFunction(localParameters);
                        BigDecimal accuracy = BigDecimal.valueOf(Math.abs(cost - previousValues.getCost()));
                        previousValues.setCost(cost);
                        previousValues.setAccuracy(accuracy);
                        System.out.println("[Optimized Gradient] Iteration " + i + ": Cost = " + previousValues.getCost() + ", Accuracy = " + previousValues.getAccuracy().toPlainString());

                        if (accuracy.compareTo(epsilon) < 0) {
                            break;
                        }
                    }
                });
            }

            executor.shutdown();
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
            return Double.NaN;
        }

        return calculateFunction(parameters);
    }

    public static double calculateFunction(double[] parameters) {
        double x = parameters[0];
        double y = parameters[1];
        double z = parameters[2];
        return Math.pow(x, 6) + Math.pow(y, 6) + Math.pow(z, 6) + Math.sin(x) + Math.cos(y) + Math.tan(z);
    }

    public static double[] calculateGradient(double[] parameters) {
        double x = parameters[0];
        double y = parameters[1];
        double z = parameters[2];
        return new double[]{
                6 * Math.pow(x, 5) + Math.cos(x),
                6 * Math.pow(y, 5) - Math.sin(y),
                6 * Math.pow(z, 5) + 1 / Math.pow(Math.cos(z), 2)
        };
    }
}