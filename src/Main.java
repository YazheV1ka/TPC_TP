import java.math.BigDecimal;
import java.util.*;
import java.util.concurrent.*;

import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class Main {
    private static final double LEARNING_RATE = 0.0001;
    private static final BigDecimal epsilon = BigDecimal.valueOf(0.001);

    private static final Lock lock = new ReentrantLock();

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
        PreviousValues previousValues = new PreviousValues();
        double[] sumOfSquaredGradients = new double[parameters.length];
        ExecutorService executor = Executors.newFixedThreadPool(threads);

        for (int i = 0; i < iterations; i++) {
            List<Callable<Double>> tasks = new ArrayList<>();

            for (int j = 0; j < parameters.length; j++) {
                int index = j;
                tasks.add(() -> {
                    double[] gradient = calculateGradient(parameters);
                    return gradient[index];
                });
            }
            tasks.add(() -> calculateFunction(parameters));

            List<CompletableFuture<Double>> futures = new ArrayList<>();

            for (Callable<Double> task : tasks) {
                futures.add(CompletableFuture.supplyAsync(() -> {
                    try {
                        return task.call();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }, executor));
            }

            try {
                CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
            } catch (Exception e) {
                e.printStackTrace();
            }

            double[] gradient = new double[parameters.length];

            for (int j = 0; j < parameters.length; j++) {
                try {
                    gradient[j] = futures.get(j).get();
                } catch (InterruptedException | ExecutionException e) {
                    e.printStackTrace();
                }
            }

            try {
                double cost = futures.get(parameters.length).get();
                lock.lock();
                try {
                    for (int j = 0; j < parameters.length; j++) {
                        sumOfSquaredGradients[j] += Math.pow(gradient[j], 2);
                        double adaptiveLearningRate = LEARNING_RATE / Math.sqrt(sumOfSquaredGradients[j]);
                        parameters[j] -= adaptiveLearningRate * gradient[j];
                    }

                    BigDecimal accuracy = BigDecimal.valueOf(Math.abs(cost - previousValues.getCost()));
                    previousValues.setCost(cost);
                    previousValues.setAccuracy(accuracy);
                    System.out.println("[Parallel Gradient] Iteration " + i + ": Cost = " + previousValues.getCost() + ", Accuracy = " + previousValues.getAccuracy().toPlainString());

                    if (accuracy.compareTo(epsilon) < 0) {
                        break;
                    }
                } finally {
                    lock.unlock();
                }
            } catch (InterruptedException | ExecutionException e) {
                e.printStackTrace();
            }
        }

        executor.shutdown();
        return calculateFunction(parameters);
    }

    public static double runOptimizedParallelGradientDescent(double[] parameters, int iterations, int threads) {
        PreviousValues previousValues = new PreviousValues();
        double[] sumOfSquaredGradients = new double[parameters.length];
        ArrayBlockingQueue<Runnable> taskQueue = new ArrayBlockingQueue<>(threads * 2);
        ThreadPoolExecutor executor = new ThreadPoolExecutor(threads, threads, 0L, TimeUnit.MILLISECONDS, taskQueue, new ThreadPoolExecutor.CallerRunsPolicy());
        executor.prestartAllCoreThreads();
        executor.setMaximumPoolSize(threads);

        for (int i = 0; i < iterations; i++) {
            CountDownLatch latch = new CountDownLatch(parameters.length + 1);
            double[] gradient = new double[parameters.length];
            AtomicReference<Double> cost = new AtomicReference<>((double) 0);

            for (int j = 0; j < parameters.length; j++) {
                int index = j;
                executor.execute(() -> {
                    double[] threadGradient = calculateGradient(parameters);
                    gradient[index] = threadGradient[index];
                    latch.countDown();
                });
            }

            executor.execute(() -> {
                cost.set(calculateFunction(parameters));
                latch.countDown();
            });

            try {
                latch.await();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }

            lock.lock();
            try {
                for (int j = 0; j < parameters.length; j++) {
                    sumOfSquaredGradients[j] += Math.pow(gradient[j], 2);
                    double adaptiveLearningRate = LEARNING_RATE / Math.sqrt(sumOfSquaredGradients[j]);
                    parameters[j] -= adaptiveLearningRate * gradient[j];
                }

                BigDecimal accuracy = BigDecimal.valueOf(Math.abs(cost.get() - previousValues.getCost()));
                previousValues.setCost(cost.get());
                previousValues.setAccuracy(accuracy);
                System.out.println("[Optimized Parallel Gradient] Iteration " + i + ": Cost = " + previousValues.getCost() + ", Accuracy = " + previousValues.getAccuracy().toPlainString());

                if (accuracy.compareTo(epsilon) < 0) {
                    break;
                }
            } finally {
                lock.unlock();
            }
        }

        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
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