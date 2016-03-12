import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartUtilities;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.CategoryDataset;
import org.jfree.data.category.DefaultCategoryDataset;
import weka.classifiers.Evaluation;
import weka.classifiers.functions.LibSVM;
import weka.classifiers.trees.J48;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import java.io.*;
import java.util.Arrays;

public class Experimenter {
    static int numberOfDataSets = 10;
    //name of Experiment: tcas, Scan, ...
    static String experimentName = "tcas";

    public static void main(String[] args) throws Exception {
        /*
         * Bayad har dataset ba dataset ham name khodesh be surate crossValidation Evaluate beshe va age ham nam
         * nabashan, bayad modelEvaluation anjam bedim. baraye mesal data[1] ba data[1] bayad crossValidation anjam bede
         * va ba data[2] bayad modelValidation anjam bede.
         */

        int countOfSampling = 6;
        Instances trainingData;
        J48 j48Model;
        //LibSVM svmModel;
        //Evaluation svmEvaluation;
        Evaluation j48Evaluation;

        ExperimentalInstances data = readInstance();

        double[][] rootMeanSquaredErrorResults = new double[countOfSampling][data.getTrainSets().length];
        double[][][] rootMeanSquaredErrorResultsAll = new double[data.getTrainSets().length][countOfSampling][data.getTestSets().length];
        double[][] failFMeasureResults = new double[countOfSampling][data.getTrainSets().length];
        double[][] passFMeasureResults = new double[countOfSampling][data.getTrainSets().length];

        for (int i = 0; i < data.getTrainSets().length; i++) {
            trainingData = resampling(data.getTrainSet(i), 100);
            for (int j = 0; j < countOfSampling; j++) {
                j48Model = j48Classification(trainingData);
                //svmModel = libSVMClassification(trainingData);

                for (int k = 0; k < data.getTestSets().length; k++) {
                    j48Evaluation = new Evaluation(trainingData);
                    //svmEvaluation = new Evaluation(trainingData);

                    //Experiment for section 2
                    j48Evaluation.evaluateModel(j48Model, data.getTestSet(k));
                    //svmEvaluation.evaluateModel(svmModel, dataArray[k]);

                    //crossValidation Mode for section 1
                    /*j48Evaluation.crossValidateModel(j48Model, trainingData, 10, new Random(1));
                    //svmEvaluation.crossValidateModel(svmModel, trainingData, 10, new Random(1));*/

                    failFMeasureResults[j][k] = j48Evaluation.fMeasure(trainingData.classAttribute().indexOfValue("Fail"));
                    passFMeasureResults[j][k] = j48Evaluation.fMeasure(trainingData.classAttribute().indexOfValue("Pass"));
                    rootMeanSquaredErrorResults[j][k] = j48Evaluation.rootMeanSquaredError();
                    rootMeanSquaredErrorResultsAll[i][j][k] = j48Evaluation.rootMeanSquaredError();

                    //makeOutputResultFile(j48Evaluation, "j48", trainingData.numInstances(), (i + 1) * 10, (k + 1) * 10);
                    //makeOutputResultFile(svmEvaluation, "svm", trainingData.numInstances(), (i + 1) * 10, (k + 1) * 10);
                }
                trainingData = resampling(trainingData, 50);
            }
            //createChart(rootMeanSquaredErrorResults, (i + 1) * 10, "j48", "Root Mean Squared Error");
        }
        createChart(createAverageChartDatasets(firstAxisMain(rootMeanSquaredErrorResultsAll), trainProportionNames(data.getTrainSets().length))
                , "Root Mean Squared Error", "TrainingSet Proportion");
        createChart(createAverageChartDatasets(secondAxisMain(rootMeanSquaredErrorResultsAll), datasetSizeNames(countOfSampling))
                , "Root Mean Squared Error", "Dataset Count");
        createChart(createAverageChartDatasets(thirdAxisMain(rootMeanSquaredErrorResultsAll), testProportionNames(data.getTestSets().length))
                , "Root Mean Squared Error", "TestingSet Proportion");
    }

    public static void makeOutputResultFile(Evaluation evaluation, String name, int numInstances, int trainProportion, int testProportion) throws Exception {
        BufferedWriter writer = new BufferedWriter(new FileWriter(experimentName + "_algorithmName" + name.toUpperCase()
                + "_trainProportion" + trainProportion + "_instancesSize" + numInstances + "_testProportion" + testProportion));
        writer.write(evaluation.toSummaryString() + "\n" + evaluation.toMatrixString() + "\n" + evaluation.toClassDetailsString());
        writer.close();
    }

    public static void createChart(double[][] resultArray, int trainProportion, String modelName, String resultName) throws IOException {
        CategoryDataset dataset = createChartDataset(resultArray);
        JFreeChart chart = ChartFactory.createBarChart(
                trainProportion + "% Train Proportion for " + experimentName + " Benchmark",        // chart title
                "Dataset Count",               // domain axis label
                resultName,                  // range axis label
                dataset,                 // data
                PlotOrientation.VERTICAL,
                true,                     // include legend
                true,                     // tooltips?
                false                     // URL generator?  Not required...
        );

        int width = 800; /* Width of the image */
        int height = 800; /* Height of the image */
        File picFile = new File(modelName + "_trainProportion" + trainProportion + ".jpeg" );
        ChartUtilities.saveChartAsJPEG(picFile , chart , width , height);
    }

    public static void createChart(CategoryDataset dataset, String resultName, String X_AxisName) throws IOException {
        JFreeChart chart = ChartFactory.createLineChart(
                experimentName + " Benchmark",        // chart title
                X_AxisName,               // domain axis label
                resultName,                  // range axis label
                dataset,                 // data
                PlotOrientation.VERTICAL,
                false,                     // include legend
                true,                     // tooltips?
                false                     // URL generator?  Not required...
        );

        int width = 400; /* Width of the image */
        int height = 400; /* Height of the image */
        File picFile = new File("Average " +  X_AxisName + ".jpeg" );
        ChartUtilities.saveChartAsJPEG(picFile , chart , width , height);
    }

    public static CategoryDataset createChartDataset(double[][] results) {

        // row keys...
        String[] proportion = testProportionNames(results[0].length);

        // column keys...
        String[] datasetSizes = datasetSizeNames(results.length);

        // create the dataset...
        final DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        for (int i = 0; i < proportion.length; i++) {
            for (int j = 0; j < datasetSizes.length; j++) {
                //dataset.addValue(results[sampleIndex][testIndex], proportion, sampleSize);
                dataset.addValue(results[j][i], proportion[i], datasetSizes[j]);
            }
        }

        return dataset;
    }

    public static CategoryDataset createAverageChartDatasets(double[] result, String[] X_AxisNames) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        for (int i = 0; i < result.length; i++) {
            dataset.addValue(result[i], "", X_AxisNames[i]);
        }

        return dataset;
    }

    public static CategoryDataset[] createLineChartDatasets(double[][][] results) {
        final DefaultCategoryDataset[] dataset = new DefaultCategoryDataset[3];
        for(int i = 0; i < 3; i++) {
            dataset[i] = new DefaultCategoryDataset();
        }

        // row keys...
        //String[] trainProportion = new String[results.length];
        String[] trainProportion = trainProportionNames(results.length);

        // column keys...
        //String[] datasetSizes = new String[results[0].length];
        String[] datasetSizes = datasetSizeNames(results[0].length);

        //String[] testProportion = new String[results[0][0].length];
        String[] testProportion = testProportionNames(results[0][0].length);

        for (int i = 0; i < trainProportion.length; i++) {
            for (int j = 0; j < testProportion.length; j++) {
                for (int k = 0; k < datasetSizes.length; k++) {
                    //dataset[0].addValue(results[i][k][j], );
                }
            }
        }

        return dataset;
    }

    public static String[] trainProportionNames(int arrayLength) {
        /*proportion1 = "10%"   proportion2 = "20%"   proportion3 = "30%"   proportion4 = "40%"   proportion5 = "50%"*/
        String[] trainProportion = new String[arrayLength];
        for (int i = 0; i < trainProportion.length; i++) {
            trainProportion[i] = String.valueOf(i + 1) + "0%";
        }

        return trainProportion;
    }

    public static String[] testProportionNames(int arrayLength) {
        /*proportion1 = "10%"   proportion2 = "20%"   proportion3 = "30%"   proportion4 = "40%"   proportion5 = "50%"*/
        String[] testProportion = new String[arrayLength];
        for (int i = 0; i < testProportion.length; i++) {
            testProportion[i] = String.valueOf(i + 1) + "0%";
        }

        return testProportion;
    }

    public static String[] datasetSizeNames(int arrayLength) {
        /*sampleSizes1 = "3000"   sampleSizes2 = "1500"   sampleSizes3 = "750"   sampleSizes4 = "375"
            sampleSizes5 = "187"   sampleSizes6 = "93"*/
        String[] datasetSizes = new String[arrayLength];
        int startDatasetSize = 3000;

        for (int i = 0; i < datasetSizes.length; i++) {
            datasetSizes[i] = String.valueOf(startDatasetSize);
            startDatasetSize /= 2;
        }

        return datasetSizes;
    }

    public static double[] firstAxisMain(double[][][] result) {
        double[] data = new double[result.length];

        for (int i = 0; i < result.length; i++) {
            double sum = 0;
            for (int j = 0; j < result[0].length; j++) {
                for (int k = 0; k < result[0][0].length; k++) {
                    sum += result[i][j][k];
                }
            }
            data[i] = sum / (result[0].length * result[0][0].length);
        }

        return data;
    }

    public static double[] secondAxisMain(double[][][] result) {
        double[] data = new double[result[0].length];

        for (int i = 0; i < result[0].length; i++) {
            double sum = 0;
            for (int j = 0; j < result.length; j++) {
                for (int k = 0; k < result[0][0].length; k++) {
                    sum += result[j][i][k];
                }
            }
            data[i] = sum / (result.length * result[0][0].length);
        }

        return data;
    }

    public static double[] thirdAxisMain(double[][][] result) {
        double[] data = new double[result[0][0].length];

        for (int i = 0; i < result[0][0].length; i++) {
            double sum = 0;
            for (int j = 0; j < result.length; j++) {
                for (int k = 0; k < result[0].length; k++) {
                    sum += result[j][k][i];
                }
            }
            data[i] = sum / (result.length * result[0].length);
        }

        return data;
    }

    public static ExperimentalInstances readInstance() throws IOException {

        //FileHaye arff dakhele folder bayad be shekle data0.arff ta data9.arff bashad
        ExperimentalInstances data = new ExperimentalInstances();

        Instances[] dataArray = new Instances[numberOfDataSets];

        for (int i = 0; i < numberOfDataSets; i++) {
            BufferedReader reader = new BufferedReader(
                    new FileReader("data" + i + ".arff"));

            dataArray[i] = new Instances(reader);
            reader.close();
            // setting class attribute
            dataArray[i].setClassIndex(dataArray[i].numAttributes() - 1);
        }

        data.setDataSet(dataArray);

        return data;
    }

    public static Instances resampling(Instances dataSet, double samplingPercentage) throws Exception {
        Resample filter = new Resample();
        filter.setSampleSizePercent(samplingPercentage);
        filter.setInputFormat(dataSet);
        return Filter.useFilter(dataSet, filter);
    }

    public static J48 j48Classification(Instances trainSet) throws Exception {
        J48 model = new J48();
        String[] options = weka.core.Utils.splitOptions("-C 0.25 -M 2");
        model.setOptions(options);

        model.buildClassifier(trainSet);
        return model;
    }

    public static LibSVM libSVMClassification(Instances trainSet) throws Exception {
        LibSVM model = new LibSVM();
        String[] options = weka.core.Utils.splitOptions("-S 0 -K 2 -D 3 -G 0.0 -R 0.0 -N 0.5 -M 40.0 -C 1.0 -E 0.001 -P 0.1 -seed 1");
        model.setOptions(options);

        model.buildClassifier(trainSet);
        return model;
    }
}

class ExperimentalInstances{
    Instances[] trainSet;
    Instances[] testSet;

    public Instances[] getTrainSets() {
        return trainSet;
    }

    public Instances[] getTestSets() {
        return testSet;
    }

    public Instances getTrainSet(int index) {
        return trainSet[index];
    }

    public Instances getTestSet(int index) {
        return testSet[index];
    }

    public void setTrainSets(Instances[] trainSet) {
        this.trainSet = trainSet;
    }

    public void setTestSets(Instances[] testSet) {
        this.testSet = testSet;
    }

    public void setTrainSet(Instances instances, int index) {
        this.trainSet[index] = instances;
    }

    public void setTestSet(Instances instances, int index) {
        this.testSet[index] = instances;
    }

    public void setDataSet(Instances[] dataArray) {
        setTrainSets(Arrays.copyOfRange(dataArray, 0, (dataArray.length / 2)));
        setTestSets(Arrays.copyOfRange(dataArray, (dataArray.length / 2),  dataArray.length));
    }
}