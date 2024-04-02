import java.math.BigDecimal;

class PreviousValues {
    double cost = 0;
    BigDecimal accuracy = BigDecimal.valueOf(0);

    public double getCost() {
        return cost;
    }

    public void setCost(double cost) {
        this.cost = cost;
    }

    public BigDecimal getAccuracy() {
        return accuracy;
    }

    public void setAccuracy(BigDecimal accuracy) {
        this.accuracy = accuracy;
    }
}