package ML.HMM;
/**
 *
 * @author Prince Isaac
 */
import Util.Validation.Validator;
import javafx.util.Pair;

import java.util.*;

public class HMM_For_Testing {
    private String name;
    private int numberOfStates;
    private int numberOfObservations;
    private Vector<String> states;
    private Vector<String> observations;
    private Hashtable<String, Double> initialProbabilities;
    private Hashtable<Pair<String, String>, Double> transitionMatrix;
    private Hashtable<Pair<String, String>, Double> emissionMatrix;

    public HMM_For_Testing(String name, Vector<String> states, Vector<String> observations, Hashtable<String, Double> initialProbabilities, Hashtable<Pair<String, String>, Double> transitionMatrix, Hashtable<Pair<String, String>, Double> emissionMatrix) throws Exception {
        this.name = name;
        this.states = states;
        this.numberOfStates = states.size();
        this.observations = observations;
        this.numberOfObservations = observations.size();

        this.initialProbabilities = initialProbabilities;
        if (!this.validateInitialProbability(initialProbabilities))
            throw new Exception("Initial Probabilities sum must be equal 1.0");
        if (!this.validateInitialProbabilitiesAndStates(states, initialProbabilities))
            throw new Exception("States size and Initial Probabilities size must be equal");

        this.transitionMatrix = transitionMatrix;
        if (!this.validateTransitionMatrix(transitionMatrix, states))
            throw new Exception("Check the transition matrix elements");

        this.emissionMatrix = emissionMatrix;
        if (!this.validateEmissionMatrix(emissionMatrix, states, observations))
            throw new Exception("Check the emission matrix elements");
    }

    public HMM_For_Testing(String filepath) {

    }

    private boolean validateInitialProbability(Hashtable<String, Double> initialProbabilities) {
        return Validator.getInstance().summationIsOne(initialProbabilities);
    }


    private boolean validateInitialProbabilitiesAndStates(Vector<String> states, Hashtable<String, Double> initialProbabilities) {
        return Validator.getInstance().isValidInitialProbabilities(states, initialProbabilities);
    }

    private boolean validateTransitionMatrix(Hashtable<Pair<String, String>, Double> transitionMatrix, Vector<String> states) {
        return Validator.getInstance().isValidTransitionMatrix(transitionMatrix, states);
    }

    private boolean validateEmissionMatrix(Hashtable<Pair<String, String>, Double> emissionMatrix, Vector<String> states, Vector<String> observations) {
        return Validator.getInstance().isValidEmissionMatrix(emissionMatrix, states, observations);
    }

    public int getNumberOfStates() {
        return this.numberOfStates;
    }

    public Vector<String> getStates() {
        return states;
    }

    public void setNumberOfStates(int numberOfStates) {
        this.numberOfStates = numberOfStates;
    }

    public int getNumberOfObservations() {
        return numberOfObservations;
    }
    public Vector<String> getObservations() { return observations; }


    public void setNumberOfObservations(int numberOfObservations) {
        this.numberOfObservations = numberOfObservations;
    }

    public Hashtable<String, Double> getInitialProbabilities() {
        return initialProbabilities;
    }


    public void setInitialProbabilities(Hashtable<String, Double> initialProbabilities) {
        this.initialProbabilities = initialProbabilities;
    }

    public Hashtable<Pair<String, String>, Double> getTransitionMatrix() {
        return transitionMatrix;
    }

    public void setTransitionMatrix(Hashtable<Pair<String, String>, Double> transitionMatrix) {
        this.transitionMatrix = transitionMatrix;
    }

    public Hashtable<Pair<String, String>, Double> getEmissionMatrix() {
        return emissionMatrix;
    }

    public void setEmissionMatrix(Hashtable<Pair<String, String>, Double> emissionMatrix) {
        this.emissionMatrix = emissionMatrix;
    }


    public Double getTransitionValue(String firstState, String secondState) {
        return this.transitionMatrix.get(new Pair<String, String>(firstState, secondState));
    }


    public Double getEmissionValue(String state, String observation) {
        return this.emissionMatrix.get(new Pair<String, String>(state, observation));
    }


    public Double getInitialProbability(String state) {
        return this.initialProbabilities.get(state);
    }


    public double evaluateUsingBruteForce(Vector<String> states, Vector<String> observations) throws Exception {
        if (states.size() != observations.size())
            throw new Exception("States and Observations must be at a same size!");

        String previousState = "";
        double probability = 0.0;
        double result = 0.0;

        for (int i = 0; i < states.size(); i++) {
            probability = this.getInitialProbability(states.get(i));
            previousState = "";
            for (int j = 0; j < observations.size(); j++) {
                double emissionValue = this.getEmissionValue(states.get(j), observations.get(j));
                double transitionValue = 0.0;
                if (j != 0) {
                    transitionValue += this.getTransitionValue(previousState, states.get(j));
                    probability *= transitionValue * emissionValue;
                }
                previousState = states.get(j);
            }
            result += probability;
        }

        return result;
    }

    public double evaluateUsingForward_Backward(Vector<String> states, Vector<String> observations) throws Exception {
        if (observations.size() != states.size()) {
            throw new Exception("States and Observations must be at a same size");
        }

        double result = 0.0;

        Vector<Hashtable<String, Double>> alpha = this.calculateForwardProbabilities(states, observations);
        Vector<Hashtable<String, Double>> beta = this.calculateBackwardProbabilities(states, observations);
        

        for (int t = 0; t < states.size(); t++) {
            for (int i = 0; i < alpha.size(); i++) {
                result += (alpha.get(t).get(states.get(i)) * beta.get(t).get(states.get(i)));
            }
        }

        return result;
    }


    public Vector<Hashtable<String, Double>> calculateForwardProbabilities(Vector<String> states, Vector<String> observations) {
        Vector<Hashtable<String, Double>> alpha = new Vector<Hashtable<String, Double>>();
        alpha.add(new Hashtable<String, Double>());
        double sum1 = 0.0;
        for(int i = 0; i < states.size(); i++) {
            alpha.elementAt(0).put(states.get(i), this.getInitialProbability(states.get(i)) * this.getEmissionValue(states.get(i), observations.get(0)));
            sum1 += this.getInitialProbability(states.get(i)) * this.getEmissionValue(states.get(i), observations.get(0));
        }

        for(int i = 0; i < states.size(); i++) {
            alpha.elementAt(0).put(states.get(i), alpha.elementAt(0).get(states.get(i)) * (1 / sum1));
        }

        sum1 = 0.0;
        for (int t = 1; t < states.size(); t++) {
            alpha.add(new Hashtable<String, Double>());
            for (int i = 0; i < states.size(); i++) {
                double probability = 0.0;
                for (int j = 0; j < states.size(); j++) {
                    probability += alpha.elementAt(t - 1).get(states.get(j)) * this.getTransitionValue(states.get(j), states.get(i));
                }
                alpha.elementAt(t).put(states.get(i), probability * this.getEmissionValue(states.get(i), observations.get(t)));
                sum1 += probability * this.getEmissionValue(states.get(i), observations.get(t));
            }
        }

        for (int t = 1; t < states.size(); t++) {
            for (int i = 0; i < states.size(); i++) {
                alpha.elementAt(t).put(states.get(i), alpha.elementAt(t).get(states.get(i)) * (1 / sum1));
            }
        }

        return alpha;
    }
    public Vector<Hashtable<String, Double>> calculateBackwardProbabilities(Vector<String> states, Vector<String> observations) {
        Vector<Hashtable<String, Double>> beta = new Vector<Hashtable<String, Double>>();
        beta.add(new Hashtable<String, Double>());
        double sum1 = 0.0;

        for (int i = 0; i < states.size(); i++) {
            beta.elementAt(0).put(states.get(i), 1.0);
            sum1 += 1.0;
        }

        for (int i = 0; i < states.size(); i++) {
            beta.elementAt(0).put(states.get(i), beta.elementAt(0).get(states.get(i)) * (1 / sum1));
        }

        sum1 = 0.0;

        for (int t = states.size() - 2; t >= 0; t--) {
            beta.insertElementAt(new Hashtable<String, Double>(), 0);
            for (int i = 0; i < states.size(); i++) {
                double probability = 0.0;
                for (int j = 0; j < states.size(); j++) {
                    probability += beta.elementAt(1).get(states.get(j)) * this.getEmissionValue(states.get(j),
                            observations.get(t)) * this.getTransitionValue(states.get(i), states.get(j));
                }
                beta.elementAt(0).put(states.get(i), probability);
                sum1 += probability;
            }
        }

        for (int t = states.size() - 2; t >= 0; t--) {
            for (int i = 0; i < states.size(); i++) {
                beta.elementAt(0).put(states.get(i), beta.elementAt(0).get(states.get(i)) * (1 / sum1));
            }
        }

        return beta;
    }
}