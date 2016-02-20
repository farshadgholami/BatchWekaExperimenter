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
import java.util.Random;

public class Experimenter {
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

        Instances [] dataArray = readInstance();

        double[][] rootMeanSquaredErrorResults = new double[countOfSampling][dataArray.length];
        double[][] failFMeasureResults = new double[countOfSampling][dataArray.length];
        double[][] passFMeasureResults = new double[countOfSampling][dataArray.length];

        for (int i = 0; i < dataArray.length; i++) {
            trainingData = resampling(dataArray[i], 100);
            for (int j = 0; j < countOfSampling; j++) {
                j48Model = j48Classification(trainingData);
                //svmModel = libSVMClassification(trainingData);

                for (int k = 0; k < dataArray.length; k++) {
                    j48Evaluation = new Evaluation(trainingData);
                    //svmEvaluation = new Evaluation(trainingData);
                    if (k == i) {
                        j48Evaluation.crossValidateModel(j48Model, trainingData, 10, new Random(1));
                        //svmEvaluation.crossValidateModel(svmModel, trainingData, 10, new Random(1));
                    } else {
                        j48Evaluation.evaluateModel(j48Model, dataArray[k]);
                        //svmEvaluation.evaluateModel(svmModel, dataArray[k]);
                    }

                    failFMeasureResults[j][k] = j48Evaluation.fMeasure(trainingData.classAttribute().indexOfValue("Fail"));
                    passFMeasureResults[j][k] = j48Evaluation.fMeasure(trainingData.classAttribute().indexOfValue("Pass"));
                    rootMeanSquaredErrorResults[j][k] = j48Evaluation.rootMeanSquaredError();

                    makeOutputResultFile(j48Evaluation, "j48", trainingData.numInstances(), (i + 1) * 10, (k + 1) * 10);
                    //makeOutputResultFile(svmEvaluation, "svm", trainingData.numInstances(), (i + 1) * 10, (k + 1) * 10);
                }
                trainingData = resampling(trainingData, 50);
            }
            createChart(rootMeanSquaredErrorResults, (i + 1) * 10, "j48", "Root Mean Squared Error");
        }
    }

    public static void makeOutputResultFile(Evaluation evaluation, String name, int numInstances, int trainProportion, int testProportion) throws Exception {
        BufferedWriter writer = new BufferedWriter(new FileWriter(name + "_trainProportion" + trainProportion + "_instancesSize" + numInstances + "_testProportion" + testProportion));
        writer.write(evaluation.toSummaryString() + "\n" + evaluation.toMatrixString() + "\n" + evaluation.toClassDetailsString());
        writer.close();
    }

    public static void createChart(double[][] resultArray, int trainProportion, String modelName, String resultName) throws IOException {
        CategoryDataset dataset = createChartDataset(resultArray);
        JFreeChart chart = ChartFactory.createBarChart(
                trainProportion + "% Train Proportion",        // chart title
                "Dataset Count",               // domain axis label
                resultName,                  // range axis label
                dataset,                 // data
                PlotOrientation.VERTICAL,
                true,                     // include legend
                true,                     // tooltips?
                false                     // URL generator?  Not required...
        );

        int width = 800; /* Width of the image */
        int height = 1280; /* Height of the image */
        File picFile = new File(modelName + "_trainProportion" + trainProportion + ".jpeg" );
        ChartUtilities.saveChartAsJPEG(picFile , chart , width , height);
    }

    public static CategoryDataset createChartDataset(double[][] results) {

        // row keys...
        String[] proportion = new String[5];
        for (int i = 0; i < 5; i++) {
            /*proportion1 = "10%"   proportion2 = "20%"   proportion3 = "30%"   proportion4 = "40%"   proportion5 = "50%"*/
            proportion[i] = String.valueOf(i + 1) + "0%";
        }

        // column keys...
        String[] sampleSizes = new String[6];
        int sampleSize = 3000;

        for (int i = 0; i < 6; i++) {
            /*sampleSizes1 = "3000"   sampleSizes2 = "1500"   sampleSizes3 = "750"   sampleSizes4 = "375"
            sampleSizes5 = "187"   sampleSizes6 = "93"*/
            sampleSizes[i] = String.valueOf(sampleSize);
            sampleSize /= 2;
        }

        // create the dataset...
        final DefaultCategoryDataset dataset = new DefaultCategoryDataset();

        for (int i = 0; i < proportion.length; i++) {
            for (int j = 0; j < sampleSizes.length; j++) {
                //dataset.addValue(results[sampleIndex][testIndex], proportion, sampleSize);
                dataset.addValue(results[j][i], proportion[i], sampleSizes[j]);
            }
        }

        return dataset;
    }

    public static Instances[] readInstance() throws IOException {

        //FileHaye arff dakhele folder bayad be shekle data0.arff ta data4.arff bashad
        Instances[] dataArray = new Instances[5];
        for (int i = 0;i < 5; i++) {
            BufferedReader reader = new BufferedReader(
                    new FileReader("data" + i + ".arff"));
            dataArray[i] = new Instances(reader);
            reader.close();
            // setting class attribute
            dataArray[i].setClassIndex(dataArray[i].numAttributes() - 1);
        }


        return dataArray;
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