package test;

import com.sun.tools.javac.Main;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

public class TestMethods {
    private static final double LEARNING_RATE = 0.0001;

    public static double runSequentialGradientDescentAccuracy(double[] parameters, int iterations) {
        double previousCost = 0;
        double accuracy = 0;
        for (int i = 0; i < iterations; i++) {
            double[] gradient = calculateGradient(parameters);
            for (int j = 0; j < parameters.length; j++) {
                parameters[j] -= LEARNING_RATE * gradient[j];
            }
            double cost = calculateCost(parameters);
            accuracy= Math.abs(cost - previousCost);
            previousCost = cost;

        }
        System.out.println();
        return accuracy;
    }

    public static double runParallelStochasticGradientDescentAccuracy(double[] parameters, int iterations, int threads) {
        double accuracy = 0;
        ExecutorService executor = Executors.newFixedThreadPool(threads);

        for (int i = 0; i < iterations; i++) {
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
                });
            }
        }
        executor.shutdown();
        try {
            executor.awaitTermination(Long.MAX_VALUE, TimeUnit.NANOSECONDS);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }


        return accuracy;
    }

    public static double runOptimizedParallelGradientDescentAccuracy(double[] parameters, int iterations, int threads) {
        double previousCost = 0, accuracy = 0;
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
            accuracy = Math.abs(cost - previousCost);
            previousCost = cost;

        }

        return accuracy;
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
