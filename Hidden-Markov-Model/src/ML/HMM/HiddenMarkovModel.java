package ML.HMM;
/**
 *
 * @author Prince Isaac
 */
import SA.Statistics.StatisticalOperations;
import Util.Validation.Validator;
import javafx.util.Pair;
import java.util.*;

public class HiddenMarkovModel {
    private String name;
    private int numberOfStates;
    private int numberOfObservations;
    private Vector<String> states;
    private Vector<String> observations;
    private Hashtable<String, Double> initialProbabilities;
    private Hashtable<Pair<String, String>, Double> transitionMatrix;
    private Hashtable<Pair<String, String>, Double> emissionMatrix;
    private Vector<Hashtable<String, Double>> alpha;
    private Vector<Hashtable<String, Double>> beta;


    public HiddenMarkovModel(String name, Vector<String> states, Vector<String> observations, Hashtable<String, Double> initialProbabilities, Hashtable<Pair<String, String>, Double> transitionMatrix, Hashtable<Pair<String, String>, Double> emissionMatrix) throws Exception {
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

        this.alpha = new Vector<Hashtable<String, Double>>();
        this.beta = new Vector<Hashtable<String, Double>>();
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

    public String getName() {
        return this.name;
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

    public Vector<Hashtable<String, Double>> getAlpha() {
        return this.alpha;
    }

    public Vector<Hashtable<String, Double>> getBeta() {
        return this.beta;
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

    public double evaluateUsingForwardAlgorithm(Vector<String> states, Vector<String> observations) {
        this.alpha = this.calculateForwardProbabilities(states, observations);
        double res = 0.0;

        for (int i = 0; i < this.alpha.get(observations.size() - 1).size(); i++) {
            res += this.alpha.get(observations.size() - 1).get(states.get(i));
        }

        return res;
    }

    public Vector<Double> evaluateUsingForward_Backward(Vector<String> states, Vector<String> observations) throws Exception {
        Vector<Double> resultsVector = new Vector<Double>();

        this.alpha = this.calculateForwardProbabilities(states, observations);
        this.beta = this.calculateBackwardProbabilities(states, observations);
       
        for (int t = 0; t < states.size(); t++) {
            double result = 1.0;
            for (int i = 0; i < this.alpha.size(); i++) {
                result += (this.alpha.get(t).get(states.get(i)) * this.beta.get(t).get(states.get(i)));
            }
            resultsVector.add(result);
        }

        resultsVector = StatisticalOperations.getInstance().normalize(resultsVector);

        return resultsVector;
    }

    public Vector<Hashtable<String, Double>> calculateForwardProbabilities(Vector<String> states, Vector<String> observations) {
        this.alpha.add(new Hashtable<String, Double>());
        for(int i = 0; i < states.size(); i++) {
           this.alpha.elementAt(0).put(states.get(i), this.getInitialProbability(states.get(i)) * this.getEmissionValue(states.get(i), observations.get(0)));
        }

        for (int t = 1; t < observations.size(); t++) {
            this.alpha.add(new Hashtable<String, Double>());
            for (int i = 0; i < states.size(); i++) {
                double probability = 0.0;
                for (int j = 0; j < states.size(); j++) {
                    probability += this.alpha.elementAt(t - 1).get(states.get(j)) * this.getTransitionValue(states.get(j), states.get(i));
                }
                this.alpha.elementAt(t).put(states.get(i), probability * this.getEmissionValue(states.get(i), observations.get(t)));
            }
        }

        return this.alpha;
    }

    private Vector<Hashtable<String, Double>> calculateBackwardProbabilities(Vector<String> states, Vector<String> observations) {
        this.beta = new Vector<Hashtable<String, Double>>();
        this.beta.add(new Hashtable<String, Double>());

        for (int i = 0; i < states.size(); i++) {
            this.beta.elementAt(0).put(states.get(i), 1.0);
        }

        for (int t = observations.size() - 2; t >= 0; t--) {
            this.beta.insertElementAt(new Hashtable<String, Double>(), 0);
            for (int i = 0; i < states.size(); i++) {
                double probability = 0.0;
                for (int j = 0; j < states.size(); j++) {
                    probability += this.beta.elementAt(1).get(states.get(j)) * this.getTransitionValue(states.get(i), states.get(j))
                            * this.getEmissionValue(states.get(j), observations.get(t)) ;
                }
                this.beta.elementAt(0).put(states.get(i), probability);
            }
        }

        return this.beta;
    }

    public String getOptimalStateSequenceUsingViterbiAlgorithm(Vector<String> states, Vector<String> observations) {
        String path = "";
        Vector<Hashtable<String, Double>> dpTable = new Vector<Hashtable<String, Double>>();
        Hashtable<String, Double> statesProbabilities = new Hashtable<String, Double>();
        Hashtable<String, Double> priorProbabilities = new Hashtable<String, Double>();

        for (int i = 0; i < observations.size(); i++) {
            if (i == 0) {
                for (String state : states) {
                    double initialProbability = this.getInitialProbability(state);
                    double emissionProbability = this.getEmissionValue(state, observations.get(i));
                    statesProbabilities.put(state, Math.log(initialProbability) + Math.log(emissionProbability));
                }
            } else {
                for (String state : states) {
                    double emissionProbability = this.getEmissionValue(state, observations.get(i));
                    double bestProbability = -100000;

                    for (String prevState : priorProbabilities.keySet()) {
                        double transitionProbability = this.getTransitionValue(prevState, state);
                        double accumulate = priorProbabilities.get(prevState) + Math.log(emissionProbability) + Math.log(transitionProbability);

                        if (accumulate > bestProbability)
                            bestProbability = accumulate;
                    }
                    statesProbabilities.put(state, bestProbability);
                }
            }

            dpTable.add((Hashtable<String, Double>)statesProbabilities.clone());
            priorProbabilities = (Hashtable<String, Double>) statesProbabilities.clone();
        }

        Hashtable<String, Double> lastColumn = dpTable.get(dpTable.size() - 1);
        double totalCost = -1000000;

        for (String item : lastColumn.keySet()) {
            if (lastColumn.get(item) > totalCost) {
                totalCost = lastColumn.get(item);
            }
        }

        for (Hashtable<String, Double> column : dpTable) {
            double costPerColumn = -1000000;
            String targetState = "";
            for (String state : column.keySet()) {
                if (column.get(state) > costPerColumn) {
                    costPerColumn = column.get(state);
                    targetState = state;
                }
            }
            path += targetState + " -> ";
        }

        path += "END with total cost = " + totalCost;

        return path;
    }

    public void estimateParametersUsingBaumWelchAlgorithm(Vector<String> states, Vector<String> observations, boolean additiveSmoothing) {
        double smoothing = additiveSmoothing ? 1.0 : 0.0;
        this.alpha = this.calculateForwardProbabilities(states, observations);
        this.beta = this.calculateBackwardProbabilities(states, observations);
        Vector<Hashtable<String, Double>> gamma = new Vector<Hashtable<String, Double>>();

        for (int i = 0; i < observations.size(); i++) {
            gamma.add(new Hashtable<String, Double>());
            double probabilitySum = 0.0;
            for (String state : states) {
                double product = this.alpha.elementAt(i).get(state) * this.beta.elementAt(i).get(state);
                gamma.elementAt(i).put(state, product);
                probabilitySum += product;
            }
            if (probabilitySum == 0)
                continue;

            for (String state : states) {
                gamma.elementAt(i).put(state, gamma.elementAt(i).get(state) / probabilitySum);
            }
        }

        Vector<Hashtable<String, Hashtable<String, Double>>> eps = new Vector<Hashtable<String, Hashtable<String, Double>>>();

        for (int i = 0; i < observations.size() - 1; i++) {
            double probabilitySum = 0.0;
            eps.add(new Hashtable<String, Hashtable<String, Double>>());
            for (String fromState : states) {
                eps.elementAt(i).put(fromState, new Hashtable<String, Double>());
                for (String toState : states) {
                    double tempProbability = this.alpha.elementAt(i).get(fromState)
                            * this.beta.elementAt(i + 1).get(toState)
                            * this.getTransitionValue(fromState, toState)
                            * this.getEmissionValue(toState, observations.elementAt(i + 1));

                     eps.elementAt(i).get(fromState).put(toState, tempProbability);
                    probabilitySum += tempProbability;
                 }
            }

            if (probabilitySum == 0)
                continue;

            for (String from : states) {
                for (String to : states) {
                    eps.elementAt(i).get(from).put(to, eps.elementAt(i).get(from).get(to) / probabilitySum);
                }
            }
        }

        for (String state : states) {
            double updated = (gamma.elementAt(0).get(state) + smoothing) / (1 + (states.size() * smoothing));
            this.initialProbabilities.put(state, updated);

            double gammaProbabilitySum = 0.0;
            for (int i = 0; i < observations.size() - 1; i++) {
                gammaProbabilitySum += gamma.elementAt(i).get(state);
            }

            if (gammaProbabilitySum > 0) {
                double denominator = gammaProbabilitySum + smoothing * states.size();
                for (String to : states) {
                    double epsSum = 0.0;
                    for (int i = 0; i < observations.size() - 1; i++) {
                        epsSum += eps.elementAt(i).get(state).get(to);
                        this.transitionMatrix.put(new Pair<String, String>(state, to), (smoothing + epsSum) / denominator);
                    }
                }
            } else {
                for (String to : states) {
                    this.transitionMatrix.put(new Pair<String, String>(state, to), 0.0);
                }
            }

            gammaProbabilitySum = 0.0;

            for (int i = 0; i < observations.size(); i++) {
                gammaProbabilitySum += gamma.elementAt(i).get(state);
            }

            Hashtable<String, Double> emissionProbabilitySums = new Hashtable<String, Double>();

            for (String observation : this.observations) {
                emissionProbabilitySums.put(observation, 0.0);
            }

            for (int i = 0; i < observations.size(); i++) {
                emissionProbabilitySums.put(observations.elementAt(i), emissionProbabilitySums.get(observations.elementAt(i)) + gamma.elementAt(i).get(state));
            }

            if (gammaProbabilitySum > 0) {
                double denominator = gammaProbabilitySum + smoothing * observations.size();
                for (String observation : this.observations) {
                    this.emissionMatrix.put(new Pair<String, String>(state, observation), (smoothing + emissionProbabilitySums.get(observation)) / denominator);
                }
            } else {
                for (String observation : this.observations) {
                    this.emissionMatrix.put(new Pair<String, String>(state, observation), 0.0);
                }
            }
        }
    }
}
