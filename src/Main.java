import java.util.Scanner;
import java.util.concurrent.*;

public class Main {
    private static final double LEARNING_RATE = 0.0001;

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
        double resultStochasticMin = runParallelStochasticGradientDescent(parameters, iterations, threads);
        double endTimeStochasticMin = System.nanoTime();
        double timeStochastic = (endTimeStochasticMin - startTimeStochasticMin) / 1e6;

        double startTimeOptimizedMin = System.nanoTime();
        double resultOptimizedMin = runOptimizedParallelGradientDescent(parameters, iterations, threads);
        double endTimeOptimizedMin = System.nanoTime();
        double timeStochasticOptimized = (endTimeOptimizedMin - startTimeOptimizedMin) / 1e6;

        double accelerationStochasticOptimizedMin = timeStochastic / timeStochasticOptimized;

        System.out.println("Sequential Descent Result: " + resultSequentialMin);
        System.out.println("Sequential Descent Execution time: " + timeSequential + " milliseconds\n");

        System.out.println("Stochastic Parallel Descent Result: " + resultStochasticMin);
        System.out.println("Stochastic Parallel Descent Execution time: " + timeStochastic + " milliseconds\n");

        System.out.println("Optimized Parallel Descent Result: " + resultOptimizedMin);
        System.out.println("Optimized Parallel Descent Execution time: " + timeStochasticOptimized + " milliseconds");
        System.out.println("Optimized Parallel Descent Acceleration: " + accelerationStochasticOptimizedMin);
    }


    private static int getIterationsFromKeyboard(Scanner scanner) {
        System.out.print("Enter the number of iterations: ");
        return scanner.nextInt();
    }

    private static int getThreadsFromKeyboard(Scanner scanner) {
        System.out.print("Enter the number of threads: ");
        return scanner.nextInt();
    }

    private static double[] generateInitialParameters() {
        return new double[]{1, 1, 1};
    }

    public static double runSequentialGradientDescent(double[] parameters, int iterations) {
        PreviousValues previousValues = new PreviousValues();

        for (int i = 0; i < iterations; i++) {
            double[] gradient = calculateGradient(parameters);
            for (int j = 0; j < parameters.length; j++) {
                parameters[j] -= LEARNING_RATE * gradient[j];
            }
            double cost = calculateCost(parameters);
            double accuracy = Math.abs(cost - previousValues.getCost());
            previousValues.setCost(cost);
            previousValues.setAccuracy(accuracy);
        }
        System.out.println("\nSequential Gradient Iteration " + iterations + ": Cost = " + previousValues.getCost() + ", Accuracy = " + previousValues.getAccuracy());
        return calculateCost(parameters);
    }

    public static double runParallelStochasticGradientDescent(double[] parameters, int iterations, int threads) {
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        PreviousValues previousValues = new PreviousValues();

        for (int i = 0; i < iterations; i++) {
            CountDownLatch latch = new CountDownLatch(threads);
            for (int j = 0; j < threads; j++) {
                executor.execute(() -> {
                    double[] localParameters = parameters.clone();
                    double[] gradient = calculateGradient(localParameters);

                    for (int k = 0; k < localParameters.length; k++) {
                        localParameters[k] -= LEARNING_RATE * gradient[k];
                    }
                    synchronized (Main.class) {
                        System.arraycopy(localParameters, 0, parameters, 0, localParameters.length);
                    }
                    double cost = calculateCost(parameters);
                    double accuracy = Math.abs(cost - previousValues.getCost());
                    previousValues.setCost(cost);
                    previousValues.setAccuracy(accuracy);
                    latch.countDown();
                });
            }
            try {
                latch.await();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }
        executor.shutdown();
        System.out.println("Stochastic Gradient Iteration " + iterations + ": Cost = " + previousValues.getCost() + ", Accuracy = " + previousValues.getAccuracy());

        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        return calculateCost(parameters);
    }

    public static double runOptimizedParallelGradientDescent(double[] parameters, int iterations, int threads) {
        PreviousValues previousValues = new PreviousValues();

        for (int i = 0; i < iterations; i++) {
            double[] gradientSum = new double[parameters.length];
            for (int j = 0; j < threads; j++) {
                double[] gradient = calculateGradient(parameters);
                for (int k = 0; k < parameters.length; k++) {
                    gradientSum[k] += gradient[k];
                }
            }
            for (int j = 0; j < parameters.length; j++) {
                parameters[j] -= LEARNING_RATE * (gradientSum[j] / threads);
            }
            double cost = calculateCost(parameters);
            double accuracy = Math.abs(cost - previousValues.getCost());
            previousValues.setCost(cost);
            previousValues.setAccuracy(accuracy);
        }
        System.out.println("Optimized Parallel Iteration " + iterations + ": Cost = " + previousValues.getCost() + ", Accuracy = " + previousValues.getAccuracy());

        System.out.println();
        return calculateCost(parameters);
    }

    private synchronized static double calculateCost(double[] parameters) {
        double x = parameters[0];
        double y = parameters[1];
        double z = parameters[2];
        return Math.pow(x, 6) + Math.pow(y, 6)+Math.pow(z, 6)+Math.sin(x)+Math.cos(y)+Math.tan(z);
    }

    private synchronized static double[] calculateGradient(double[] parameters) {
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