import java.util.function.Function;

public class TestMethods {
    private static final double LEARNING_RATE = 0.01;

    public static void main(String[] args) {
        Function<Double, Double> function = x -> Math.pow(x, 6) + Math.sin(x);
        Function<Double, Double> gradient = x -> 6 * Math.pow(x, 5) + Math.cos(x);

        double result = runSequentialGD(gradient, LEARNING_RATE, 10, 0.001);

        System.out.println("Function: f(x) = x^6 + sin(x)");
        System.out.println("Global minimum Sequential: " + function.apply(result));
    }

    public static double runSequentialGD(Function<Double, Double> gradient, double alpha, int maxIterations, double tolerance) {
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
}