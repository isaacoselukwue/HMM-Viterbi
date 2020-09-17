package SA.Statistics;
/**
 *
 * @author Prince Isaac
 */
import java.util.Hashtable;
import java.util.Vector;

public class StatisticalOperations {
    private static StatisticalOperations ourInstance = new StatisticalOperations();

    public static StatisticalOperations getInstance() {
        return ourInstance;
    }

    private StatisticalOperations() {
    }
    public Vector<Hashtable<String, Double>> normalize(Vector<Hashtable<String, Double>> probabilities, Vector<String> states) {
        double sum = 0.0;
        if (states.size() == 1) return probabilities;

        for (int t = 0; t < states.size(); t++) {
            for (int i = 0; i < probabilities.size(); i++) {
                sum += (probabilities.get(t).get(states.get(i)));
            }
        }

        for (int t = 0; t < states.size(); t++) {
            for (int i = 0; i < probabilities.size(); i++) {
                double current = (probabilities.get(t).get(states.get(i)));
                probabilities.elementAt(t).put(states.get(i), current / sum);
            }
        }

        return probabilities;
    }

    public Vector<Double> normalize(Vector<Double> data) {
        Vector<Double> res = new Vector<Double>();
        double sum = 0.0;

        for (int i = 0; i < data.size(); i++) {
            sum += data.get(i);
        }

        for (int i = 0; i < data.size(); i++) {
             res.add(data.get(i) / sum);
        }

        return res;
    }
}