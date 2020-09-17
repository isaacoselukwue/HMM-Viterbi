package DA.Processing;
/**
 *
 * @author Prince Isaac
 */
import javafx.util.Pair;
import javax.swing.text.rtf.RTFEditorKit;
import java.util.Collections;
import java.util.Hashtable;
import java.util.Vector;

public class DataDecoding {
    private final String OUTER_SPLITTER = ", ";
    private final String INNER_SPLITTER = "->";
    private static DataDecoding ourInstance = new DataDecoding();

    public static DataDecoding getInstance() {
        return ourInstance;
    }

    private DataDecoding() {
    }

    public String getModelName(String nameInJson) {
        return nameInJson;
    }


    public String getModelCreationDate(String dateInJson) {
        return dateInJson;
    }

    public String getModelCreationPurpose(String purposeInJson) {
        return purposeInJson;
    }

    public Vector<String> getStates(String statesInJson) {
        Vector<String> states = new Vector<String>();
        String[] statesArray = statesInJson.split(OUTER_SPLITTER);

        Collections.addAll(states, statesArray);

        return states;
    }

    public Hashtable<String, Double> getInitialProbabilities(String initialProbabilitiesInJson) {
        Hashtable<String, Double> initialProbabilities = new Hashtable<String, Double>();

        String[] initialProb = initialProbabilitiesInJson.split(OUTER_SPLITTER);

        for (String expression : initialProb) {
            String[] tempExpression = expression.split(INNER_SPLITTER);

            for (int i = 0; i < tempExpression.length; i += 2) {
                initialProbabilities.put(tempExpression[i], Double.parseDouble(tempExpression[i + 1]));
            }
        }

        return initialProbabilities;
    }

    public Vector<String> getObservations(String observationsInJson) {
        Vector<String> observations = new Vector<String>();
        String[] expressionArray = observationsInJson.split(OUTER_SPLITTER);

        Collections.addAll(observations, expressionArray);

        return observations;
    }

    public Hashtable<Pair<String, String>, Double> getTransitionMatrix(String transitionMatrixInJson) {
        Hashtable<Pair<String, String>, Double> transitionMatrix = new Hashtable<Pair<String, String>, Double>();
        String[] tempExpressionArray = transitionMatrixInJson.split(OUTER_SPLITTER);

        for (String expression : tempExpressionArray) {
            String[] transitionExpression = expression.split(INNER_SPLITTER);

            for (int i = 0; i < transitionExpression.length; i += 3) {
                transitionMatrix.put(new Pair<String, String>(transitionExpression[i], transitionExpression[i + 1]),
                        Double.parseDouble(transitionExpression[i + 2]));
            }
        }

        return transitionMatrix;
    }

    public Hashtable<Pair<String, String>, Double> getEmissionMatrix(String emissionMatrixInJson) {
        Hashtable<Pair<String, String>, Double> emissionMatrix = new Hashtable<Pair<String, String>, Double>();
        String[] tempExpressionArray = emissionMatrixInJson.split(OUTER_SPLITTER);

        for (String expression : tempExpressionArray) {
            String[] emissionExpression = expression.split(INNER_SPLITTER);

            for (int i = 0; i < emissionExpression.length; i += 3) {
                emissionMatrix.put(new Pair<String, String>(emissionExpression[i], emissionExpression[i + 1]),
                        Double.parseDouble(emissionExpression[i + 2]));
            }
        }

        return emissionMatrix;
    }
}